import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api, ApiError, type Collection } from "../api";
import { useAuth } from "../auth";
import { Spinner } from "../ui";

// Presets = raccourcis de configuration à la création.
const PRESETS = [
  { id: "balanced", label: "Équilibré", desc: "bge-m3 + rerank ONNX", body: { rerank_enabled: true } },
  { id: "fast", label: "Rapide", desc: "sans rerank (latence min.)", body: { rerank_enabled: false } },
];

export default function Collections() {
  const { user } = useAuth();
  const [cols, setCols] = useState<Collection[] | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const isAdmin = user?.role === "admin";

  const load = () => api.listCollections().then(setCols);
  useEffect(() => {
    load();
  }, []);

  return (
    <div>
      <div className="mb-5 flex items-center justify-between">
        <h1 className="text-xl font-semibold">Collections</h1>
        {isAdmin && (
          <button className="btn-primary" onClick={() => setShowCreate(true)}>
            + Nouvelle collection
          </button>
        )}
      </div>

      {cols === null ? (
        <Spinner label="Chargement des collections…" />
      ) : cols.length === 0 ? (
        <div className="card p-8 text-center text-slate-500">
          Aucune collection.{" "}
          {isAdmin ? "Créez-en une pour commencer." : "Demandez à un admin d'en créer une."}
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {cols.map((c) => (
            <CollectionCard key={c.name} col={c} isAdmin={isAdmin} onChanged={load} />
          ))}
        </div>
      )}

      {showCreate && (
        <CreateModal
          onClose={() => setShowCreate(false)}
          onCreated={() => {
            setShowCreate(false);
            load();
          }}
        />
      )}
    </div>
  );
}

function CollectionCard({
  col,
  isAdmin,
  onChanged,
}: {
  col: Collection;
  isAdmin: boolean;
  onChanged: () => void;
}) {
  const [deleting, setDeleting] = useState(false);

  const del = async () => {
    if (!confirm(`Supprimer la collection « ${col.name} » et tous ses documents ?`)) return;
    setDeleting(true);
    try {
      await api.deleteCollection(col.name);
      onChanged();
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="card flex flex-col p-4">
      <Link to={`/c/${col.name}`} className="group">
        <h2 className="font-semibold text-slate-800 group-hover:text-brand-700">{col.name}</h2>
        <p className="mt-1 line-clamp-2 min-h-[2.5rem] text-sm text-slate-500">
          {col.description || "—"}
        </p>
      </Link>
      <div className="mt-3 flex flex-wrap gap-1 text-xs text-slate-500">
        <span className="badge bg-slate-100">{col.n_documents} docs</span>
        <span className="badge bg-slate-100">{col.n_chunks} chunks</span>
        <span className="badge bg-slate-100">{col.embedding_model.split("/").pop()}</span>
        {col.rerank_enabled && <span className="badge bg-brand-50 text-brand-700">rerank</span>}
      </div>
      <div className="mt-4 flex items-center gap-2 border-t border-slate-100 pt-3">
        <Link to={`/c/${col.name}/chat`} className="btn-primary flex-1 justify-center">
          Chat
        </Link>
        <Link to={`/c/${col.name}`} className="btn-ghost">
          Documents
        </Link>
        {isAdmin && (
          <button className="btn-ghost text-red-600" onClick={del} disabled={deleting} title="Supprimer">
            🗑
          </button>
        )}
      </div>
    </div>
  );
}

function CreateModal({ onClose, onCreated }: { onClose: () => void; onCreated: () => void }) {
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [preset, setPreset] = useState("balanced");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(name)) {
      setError("Nom invalide (lettres, chiffres, _ ou -, 1 à 64 caractères).");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const body = { name, description, ...PRESETS.find((p) => p.id === preset)!.body };
      await api.createCollection(body);
      onCreated();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Erreur");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 z-10 flex items-center justify-center bg-black/30 p-4" onClick={onClose}>
      <form
        onSubmit={submit}
        onClick={(e) => e.stopPropagation()}
        className="card w-full max-w-md space-y-4 p-6"
      >
        <h2 className="text-lg font-semibold">Nouvelle collection</h2>
        <div className="space-y-1">
          <label className="text-sm font-medium">Nom</label>
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} autoFocus />
        </div>
        <div className="space-y-1">
          <label className="text-sm font-medium">Description</label>
          <textarea
            className="input"
            rows={2}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>
        <div className="space-y-1">
          <label className="text-sm font-medium">Preset</label>
          <div className="grid grid-cols-2 gap-2">
            {PRESETS.map((p) => (
              <button
                type="button"
                key={p.id}
                onClick={() => setPreset(p.id)}
                className={`rounded-md border p-2 text-left text-sm ${
                  preset === p.id ? "border-brand-500 bg-brand-50" : "border-slate-200"
                }`}
              >
                <div className="font-medium">{p.label}</div>
                <div className="text-xs text-slate-500">{p.desc}</div>
              </button>
            ))}
          </div>
        </div>
        <p className="text-xs text-slate-400">
          Le modèle d'embedding (bge-m3) est figé à la création : en changer imposerait une
          réindexation complète.
        </p>
        {error && <p className="text-sm text-red-600">{error}</p>}
        <div className="flex justify-end gap-2">
          <button type="button" className="btn-ghost" onClick={onClose}>
            Annuler
          </button>
          <button className="btn-primary" disabled={busy}>
            {busy ? "Création…" : "Créer"}
          </button>
        </div>
      </form>
    </div>
  );
}
