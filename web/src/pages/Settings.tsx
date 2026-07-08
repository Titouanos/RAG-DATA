import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";

import { api, ApiError, type Collection } from "../api";
import { CollectionTabs, Spinner } from "../ui";

export default function Settings() {
  const { name = "" } = useParams();
  const [col, setCol] = useState<Collection | null>(null);
  const [form, setForm] = useState<Partial<Collection>>({});
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.getCollection(name).then((c) => {
      setCol(c);
      setForm(c);
    });
  }, [name]);

  const set = (patch: Partial<Collection>) => {
    setForm((f) => ({ ...f, ...patch }));
    setSaved(false);
  };

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const updated = await api.updateCollection(name, {
        description: form.description,
        llm_provider: form.llm_provider,
        llm_model: form.llm_model,
        top_k: form.top_k,
        rerank_k: form.rerank_k,
        rerank_enabled: form.rerank_enabled,
        rerank_model: form.rerank_model,
        system_prompt: form.system_prompt || null,
        ocr_enabled: form.ocr_enabled,
      });
      setCol(updated);
      setForm(updated);
      setSaved(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Erreur");
    } finally {
      setBusy(false);
    }
  };

  if (!col) return <Spinner label="Chargement…" />;

  return (
    <div>
      <h1 className="mb-4 text-xl font-semibold">{name}</h1>
      <CollectionTabs name={name} />

      <form onSubmit={save} className="max-w-2xl space-y-5">
        <section className="card p-4">
          <h2 className="mb-3 font-semibold">Embedding (figé)</h2>
          <p className="text-sm text-slate-500">
            Modèle : <span className="font-mono">{col.embedding_model}</span> ({col.dense_dim} dims,
            {col.supports_sparse ? " dense + sparse" : " dense"}). Changer d'embedding imposerait une
            réindexation complète de la collection — non modifiable ici.
          </p>
        </section>

        <section className="card space-y-3 p-4">
          <h2 className="font-semibold">Génération</h2>
          <Field label="Provider LLM">
            <select
              className="input"
              value={form.llm_provider}
              onChange={(e) => set({ llm_provider: e.target.value })}
            >
              {["gemini", "mistral", "anthropic", "ollama"].map((p) => (
                <option key={p} value={p}>
                  {p}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Modèle">
            <input
              className="input"
              value={form.llm_model ?? ""}
              onChange={(e) => set({ llm_model: e.target.value })}
            />
          </Field>
          <Field label="Prompt système (optionnel)">
            <textarea
              className="input font-mono text-xs"
              rows={4}
              placeholder="Laisser vide pour le prompt par défaut (honnêteté + citations)."
              value={form.system_prompt ?? ""}
              onChange={(e) => set({ system_prompt: e.target.value })}
            />
          </Field>
        </section>

        <section className="card space-y-3 p-4">
          <h2 className="font-semibold">Retrieval</h2>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={!!form.rerank_enabled}
              onChange={(e) => set({ rerank_enabled: e.target.checked })}
            />
            Reranking activé
          </label>
          <div className="grid grid-cols-2 gap-3">
            <Field label="top_k (chunks au LLM)">
              <input
                type="number"
                min={1}
                max={20}
                className="input"
                value={form.top_k ?? 5}
                onChange={(e) => set({ top_k: Number(e.target.value) })}
              />
            </Field>
            <Field label="rerank_k (candidats)">
              <input
                type="number"
                min={1}
                max={50}
                className="input"
                value={form.rerank_k ?? 10}
                onChange={(e) => set({ rerank_k: Number(e.target.value) })}
              />
            </Field>
          </div>
          <Field label="Modèle de rerank">
            <input
              className="input"
              value={form.rerank_model ?? ""}
              onChange={(e) => set({ rerank_model: e.target.value })}
            />
          </Field>
        </section>

        <section className="card space-y-2 p-4">
          <h2 className="font-semibold">OCR (PDF scannés)</h2>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={!!form.ocr_enabled}
              onChange={(e) => set({ ocr_enabled: e.target.checked })}
            />
            Activer l'OCR (fra+eng) pour les PDF sans couche texte
          </label>
          <p className="text-xs text-slate-400">
            Nécessite ocrmypdf + Tesseract côté serveur. Suggéré quand un document affiche
            « pages sans texte détectées ».
          </p>
        </section>

        {error && <p className="text-sm text-red-600">{error}</p>}
        <div className="flex items-center gap-3">
          <button className="btn-primary" disabled={busy}>
            {busy ? "Enregistrement…" : "Enregistrer"}
          </button>
          {saved && <span className="text-sm text-green-600">Enregistré ✓</span>}
        </div>
      </form>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="space-y-1">
      <label className="text-sm font-medium">{label}</label>
      {children}
    </div>
  );
}
