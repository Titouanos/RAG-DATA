"""Rate limiter pour Gemini API free tier.

Gère :
- **RPM** (requests per minute) : sliding window en mémoire, attend si nécessaire
  avant d'appeler pour ne pas dépasser la limite.
- **RPD** (requests per day) : compteur persisté sur disque qui reset à minuit PT.
  Lève une exception si la limite est atteinte.
- **Retry 429** : si une erreur RESOURCE_EXHAUSTED remonte malgré tout, on parse le
  `retryDelay` renvoyé par Gemini et on attend le bon délai avant de retenter.

Les limites par défaut correspondent au free tier 2026. Elles sont modifiables
dans config.yaml.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Limites free tier par défaut (avril 2026)
# ---------------------------------------------------------------------------


# On prend des valeurs légèrement sous les limites réelles pour avoir du jeu.
# Modifiables dans config.yaml → rate_limits.
DEFAULT_FREE_TIER_LIMITS: dict[str, dict[str, int]] = {
    "gemini-2.5-pro":         {"rpm": 4,  "rpd": 90},
    "gemini-2.5-flash":       {"rpm": 9,  "rpd": 240},
    "gemini-2.5-flash-lite":  {"rpm": 14, "rpd": 950},
    "gemini-embedding-001":   {"rpm": 5,  "rpd": 950},
    # fallback pour les modèles inconnus : conservateur
    "_default":               {"rpm": 5,  "rpd": 100},
}


# ---------------------------------------------------------------------------
# RateLimiter : une instance par projet, suit tous les modèles utilisés
# ---------------------------------------------------------------------------


@dataclass
class _ModelState:
    """État interne pour un modèle donné."""
    rpm_limit: int
    rpd_limit: int
    recent_calls: deque = field(default_factory=deque)  # timestamps des dernières minutes
    daily_count: int = 0
    daily_reset_at: float = 0.0  # timestamp de prochain reset


class RateLimiter:
    """Rate limiter thread-safe par modèle.

    Usage:
        limiter = RateLimiter(persist_path=Path("rate_state.json"))
        limiter.acquire("gemini-2.5-flash")  # attend si besoin, puis réserve 1 slot
        # ... faire l'appel API ...
    """

    def __init__(self, persist_path: Path | None = None,
                 limits_overrides: dict[str, dict[str, int]] | None = None):
        """
        :param persist_path: où sauvegarder le compteur journalier (pour survivre
                             aux redémarrages dans la même journée PT).
        :param limits_overrides: dict {model_name: {rpm, rpd}} pour surcharger
                                 les défauts.
        """
        self.persist_path = persist_path
        self._limits = dict(DEFAULT_FREE_TIER_LIMITS)
        if limits_overrides:
            for model, lim in limits_overrides.items():
                self._limits[model] = {**self._limits.get(model, DEFAULT_FREE_TIER_LIMITS["_default"]),
                                        **lim}
        self._states: dict[str, _ModelState] = {}
        self._lock = threading.Lock()
        self._load_persisted()

    # ------------------------------------------------------------------
    # API principale
    # ------------------------------------------------------------------

    def acquire(self, model: str, wait_rpm: bool = True) -> None:
        """Réserve un slot d'appel pour le modèle.

        - Vérifie le quota journalier (RPD) et lève si épuisé.
        - Attend si nécessaire pour respecter la limite par minute (RPM).
        - Incrémente les compteurs.
        """
        with self._lock:
            state = self._get_state(model)
            self._maybe_reset_daily(state)

            # 1. RPD : hard stop si atteint (rpd_limit <= 0 signifie "pas de limite RPD")
            if state.rpd_limit > 0 and state.daily_count >= state.rpd_limit:
                remaining = state.daily_reset_at - time.time()
                hours = max(0, remaining / 3600)
                raise QuotaExhaustedError(
                    f"Quota journalier (RPD) épuisé pour le modèle '{model}' : "
                    f"{state.daily_count}/{state.rpd_limit} requêtes utilisées. "
                    f"Reset dans ~{hours:.1f}h (minuit Pacific Time)."
                )

            # 2. RPM : attendre si nécessaire
            if wait_rpm:
                self._wait_for_rpm_slot(state, model)

            # 3. Réserver le slot
            now = time.time()
            state.recent_calls.append(now)
            state.daily_count += 1

        self._persist()

    def report_429(self, model: str, retry_delay_seconds: float | None = None):
        """Notifie le limiter qu'un 429 a été reçu.

        On réserve 'défensivement' un délai additionnel pour ne pas retenter trop vite.
        """
        with self._lock:
            state = self._get_state(model)
            # Marque la minute courante comme 'remplie' pour éviter de retenter
            # immédiatement : on ajoute des timestamps bidons pour forcer l'attente.
            now = time.time()
            # Supprime les anciens > 60s
            while state.recent_calls and now - state.recent_calls[0] > 60:
                state.recent_calls.popleft()
            # Pad jusqu'à atteindre la limite RPM
            while len(state.recent_calls) < state.rpm_limit:
                state.recent_calls.append(now)

        if retry_delay_seconds:
            logger.warning("429 sur %s : le serveur demande d'attendre %.1fs",
                           model, retry_delay_seconds)

    def stats(self) -> dict[str, dict[str, int]]:
        """Retourne l'état actuel pour affichage."""
        out = {}
        with self._lock:
            now = time.time()
            for model, state in self._states.items():
                self._maybe_reset_daily(state)
                # Nettoie la fenêtre RPM
                while state.recent_calls and now - state.recent_calls[0] > 60:
                    state.recent_calls.popleft()
                out[model] = {
                    "rpm_used": len(state.recent_calls),
                    "rpm_limit": state.rpm_limit,
                    "rpd_used": state.daily_count,
                    "rpd_limit": state.rpd_limit,
                    "rpd_reset_in_seconds": max(0, int(state.daily_reset_at - now)),
                }
        return out

    # ------------------------------------------------------------------
    # Internes
    # ------------------------------------------------------------------

    def _get_state(self, model: str) -> _ModelState:
        if model not in self._states:
            limits = self._limits.get(model) or self._limits["_default"]
            self._states[model] = _ModelState(
                rpm_limit=limits["rpm"],
                rpd_limit=limits["rpd"],
                daily_reset_at=_next_pt_midnight_ts(),
            )
        return self._states[model]

    def _maybe_reset_daily(self, state: _ModelState):
        """Reset le compteur RPD si minuit PT est passé."""
        now = time.time()
        if now >= state.daily_reset_at:
            state.daily_count = 0
            state.daily_reset_at = _next_pt_midnight_ts()

    def _wait_for_rpm_slot(self, state: _ModelState, model: str):
        """Attend qu'un slot RPM se libère."""
        while True:
            now = time.time()
            # Nettoie la fenêtre : supprime les timestamps > 60s
            while state.recent_calls and now - state.recent_calls[0] > 60:
                state.recent_calls.popleft()

            if len(state.recent_calls) < state.rpm_limit:
                return  # slot disponible

            # Attendre jusqu'à ce que le plus ancien timestamp dépasse 60s
            oldest = state.recent_calls[0]
            wait = 60 - (now - oldest) + 0.2  # petite marge
            logger.info("Rate limit atteint sur %s (%d/%d RPM). Attente de %.1fs...",
                        model, len(state.recent_calls), state.rpm_limit, wait)
            # Releaser le lock pendant l'attente pour ne pas bloquer les autres threads
            self._lock.release()
            try:
                time.sleep(wait)
            finally:
                self._lock.acquire()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_persisted(self):
        """Charge le compteur journalier depuis le disque."""
        if not self.persist_path or not self.persist_path.exists():
            return
        try:
            data = json.loads(self.persist_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Impossible de charger %s : %s", self.persist_path, exc)
            return

        now = time.time()
        for model, entry in (data.get("states") or {}).items():
            reset_at = float(entry.get("daily_reset_at", 0))
            # Si le reset est passé, on ignore (compteur repart à zéro)
            if reset_at <= now:
                continue
            limits = self._limits.get(model) or self._limits["_default"]
            self._states[model] = _ModelState(
                rpm_limit=limits["rpm"],
                rpd_limit=limits["rpd"],
                daily_count=int(entry.get("daily_count", 0)),
                daily_reset_at=reset_at,
            )

    def _persist(self):
        """Sauvegarde les compteurs sur disque."""
        if not self.persist_path:
            return
        try:
            data = {
                "states": {
                    model: {
                        "daily_count": state.daily_count,
                        "daily_reset_at": state.daily_reset_at,
                    }
                    for model, state in self._states.items()
                }
            }
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            self.persist_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Échec sauvegarde rate limiter : %s", exc)


class QuotaExhaustedError(RuntimeError):
    """Le quota journalier (RPD) est atteint pour un modèle."""


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------


def _next_pt_midnight_ts() -> float:
    """Retourne le timestamp du prochain minuit Pacific Time.

    PT = UTC-8 en hiver (PST), UTC-7 en été (PDT). On utilise une approximation
    simple : UTC-8 (pire cas, reset un peu plus tard que minuit PT réel l'été).
    C'est suffisant pour du rate limiting défensif.
    """
    now_utc = datetime.now(timezone.utc)
    # Midnight PT = 08:00 UTC (PST) ou 07:00 UTC (PDT). On prend 08:00 pour être safe.
    today_reset = now_utc.replace(hour=8, minute=0, second=0, microsecond=0)
    if now_utc >= today_reset:
        today_reset += timedelta(days=1)
    return today_reset.timestamp()


# ---------------------------------------------------------------------------
# Parser du retry_delay renvoyé par Gemini dans les erreurs 429
# ---------------------------------------------------------------------------


RETRY_DELAY_RE = re.compile(r"retry[_\-\s]*delay[\"':\s]+(?:\{[^}]*seconds[\"':\s]+)?(\d+)",
                             re.IGNORECASE)


def extract_retry_delay(exc: Exception) -> float | None:
    """Essaie d'extraire le retryDelay (en secondes) d'une erreur 429 Gemini.

    Le SDK remonte une erreur avec un message qui contient typiquement :
        { 'error': { ... 'details': [ { '@type': 'type.googleapis.com/google.rpc.RetryInfo',
                                         'retryDelay': '42s' } ] } }
    """
    msg = str(exc)
    m = RETRY_DELAY_RE.search(msg)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def is_quota_error(exc: Exception) -> bool:
    """True si l'exception ressemble à un 429 RESOURCE_EXHAUSTED."""
    msg = str(exc).lower()
    return any(s in msg for s in (
        "429", "resource_exhausted", "quota", "rate limit",
    ))


# ---------------------------------------------------------------------------
# Wrapper générique pour retry sur 429
# ---------------------------------------------------------------------------


def with_rate_limit_retry(
    fn: Callable,
    *,
    limiter: RateLimiter,
    model: str,
    max_attempts: int = 5,
    default_wait: float = 30.0,
):
    """Appelle fn() en respectant le rate limit et en retry sur 429.

    - Avant l'appel : acquire() pour respecter RPM/RPD proactivement.
    - En cas de 429 : parse retry_delay, attend, réessaie.
    - Les erreurs autres que 429 remontent directement.
    """
    for attempt in range(1, max_attempts + 1):
        # Acquire : attente proactive pour respecter le RPM
        limiter.acquire(model)

        try:
            return fn()
        except QuotaExhaustedError:
            raise
        except Exception as exc:
            if not is_quota_error(exc):
                raise
            # 429 : on a dépassé malgré tout (ou la limite réelle est plus basse)
            retry_delay = extract_retry_delay(exc)
            wait = retry_delay if retry_delay else default_wait * attempt
            logger.warning(
                "429 sur %s (tentative %d/%d). Attente de %.1fs avant retry...",
                model, attempt, max_attempts, wait,
            )
            limiter.report_429(model, retry_delay_seconds=retry_delay)
            if attempt >= max_attempts:
                raise
            time.sleep(wait)