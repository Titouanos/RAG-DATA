import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

import { api, type Collection, type DocumentItem, type Job } from "../api";
import { CollectionTabs, formatBytes, formatDate, Spinner, StatusBadge } from "../ui";

const TERMINAL = ["succeeded", "failed", "skipped"];

export default function CollectionDetail() {
  const { name = "" } = useParams();
  const [col, setCol] = useState<Collection | null>(null);
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const reindexInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    const [c, d, j] = await Promise.all([
      api.getCollection(name),
      api.listDocuments(name),
      api.listJobs(name),
    ]);
    setCol(c);
    setDocs(d);
    setJobs(j);
  }, [name]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Polling temps réel tant qu'un job n'est pas terminé.
  useEffect(() => {
    const active = jobs.some((j) => !TERMINAL.includes(j.status));
    if (!active) return;
    const t = setInterval(refresh, 1200);
    return () => clearInterval(t);
  }, [jobs, refresh]);

  const upload = async (files: FileList | File[]) => {
    const arr = Array.from(files);
    if (!arr.length) return;
    setUploading(true);
    try {
      await api.uploadDocuments(name, arr);
      await refresh();
    } finally {
      setUploading(false);
    }
  };

  const del = async (doc: DocumentItem) => {
    if (!confirm(`Supprimer « ${doc.source_name} » ? Ses extraits disparaîtront des réponses.`)) return;
    await api.deleteDocument(name, doc.doc_id);
    await refresh();
  };

  const activeJobs = jobs.filter((j) => !TERMINAL.includes(j.status));

  if (!col) return <Spinner label="Chargement…" />;

  return (
    <div>
      <div className="mb-1 flex items-center gap-2 text-sm text-slate-400">
        <Link to="/" className="hover:text-slate-600">
          Collections
        </Link>
        <span>/</span>
        <span className="text-slate-600">{name}</span>
      </div>
      <h1 className="mb-4 text-xl font-semibold">{name}</h1>
      <CollectionTabs name={name} />

      {/* Zone d'upload drag & drop */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          upload(e.dataTransfer.files);
        }}
        onClick={() => fileInput.current?.click()}
        className={`mb-4 cursor-pointer rounded-lg border-2 border-dashed p-6 text-center text-sm transition ${
          dragOver ? "border-brand-500 bg-brand-50" : "border-slate-300 text-slate-500"
        }`}
      >
        <input
          ref={fileInput}
          type="file"
          multiple
          hidden
          onChange={(e) => e.target.files && upload(e.target.files)}
        />
        {uploading ? "Envoi en cours…" : "Glissez des fichiers ici, ou cliquez pour parcourir"}
        <div className="mt-1 text-xs text-slate-400">PDF, Office, HTML, MindManager — max 100 Mo</div>
      </div>

      {/* Jobs en cours */}
      {activeJobs.length > 0 && (
        <div className="card mb-4 divide-y divide-slate-100">
          {activeJobs.map((j) => (
            <div key={j.id} className="p-3">
              <div className="flex items-center justify-between text-sm">
                <span className="font-medium">{j.source_name}</span>
                <span className="text-slate-500">
                  {j.stage || j.status}
                  {j.progress_total > 0 && ` ${j.progress_current}/${j.progress_total}`}
                </span>
              </div>
              <div className="mt-2 h-1.5 overflow-hidden rounded bg-slate-100">
                <div
                  className="h-full bg-brand-500 transition-all"
                  style={{
                    width: j.progress_total ? `${(j.progress_current / j.progress_total) * 100}%` : "30%",
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Tableau des documents */}
      <div className="card overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 text-left text-xs uppercase text-slate-500">
            <tr>
              <th className="px-4 py-2">Document</th>
              <th className="px-4 py-2">Statut</th>
              <th className="px-4 py-2">Chunks</th>
              <th className="px-4 py-2">Taille</th>
              <th className="px-4 py-2">Ajouté</th>
              <th className="px-4 py-2"></th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100">
            {docs.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-6 text-center text-slate-400">
                  Aucun document. Déposez-en ci-dessus.
                </td>
              </tr>
            )}
            {docs.map((d) => (
              <tr key={d.doc_id}>
                <td className="px-4 py-2">
                  <div className="font-medium text-slate-800">{d.source_name}</div>
                  {d.error_message && <div className="text-xs text-red-600">{d.error_message}</div>}
                  {d.scanned_suspect && (
                    <div className="text-xs text-amber-600">
                      ⚠ pages sans texte détectées — OCR recommandé
                    </div>
                  )}
                </td>
                <td className="px-4 py-2">
                  <StatusBadge status={d.status} />
                </td>
                <td className="px-4 py-2">{d.n_chunks}</td>
                <td className="px-4 py-2">{formatBytes(d.size_bytes)}</td>
                <td className="px-4 py-2 text-slate-500">{formatDate(d.created_at)}</td>
                <td className="px-4 py-2 text-right">
                  <button
                    className="btn-ghost"
                    title="Réindexer (re-sélectionner le fichier)"
                    onClick={() => reindexInput.current?.click()}
                  >
                    ↻
                  </button>
                  <button className="btn-ghost text-red-600" title="Supprimer" onClick={() => del(d)}>
                    🗑
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {/* Réindexation = re-upload (les fichiers sources ne sont pas conservés côté serveur). */}
      <input
        ref={reindexInput}
        type="file"
        multiple
        hidden
        onChange={(e) => e.target.files && upload(e.target.files)}
      />
      <p className="mt-2 text-xs text-slate-400">
        Réindexer : re-déposez le fichier (même contenu → ignoré, contenu modifié → mis à jour).
      </p>
    </div>
  );
}
