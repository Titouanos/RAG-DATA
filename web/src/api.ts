// Client API : fetch same-origin (proxifié vers FastAPI en dev), cookies de session inclus.

export interface User {
  id: number;
  username: string;
  role: string;
}

export interface Collection {
  name: string;
  description: string;
  embedder: string;
  embedding_model: string;
  dense_dim: number;
  supports_sparse: boolean;
  rerank_enabled: boolean;
  rerank_model: string;
  top_k: number;
  rerank_k: number;
  llm_provider: string;
  llm_model: string;
  system_prompt: string | null;
  ocr_enabled: boolean;
  n_documents: number;
  n_chunks: number;
}

export interface DocumentItem {
  doc_id: string;
  source_name: string;
  doc_type: string;
  status: string;
  n_chunks: number;
  size_bytes: number;
  scanned_suspect: boolean;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface Job {
  id: number;
  collection: string;
  type: string;
  status: string;
  source_name: string;
  doc_id: string;
  stage: string;
  progress_current: number;
  progress_total: number;
  message: string;
  created_at: string;
  updated_at: string;
}

export interface Source {
  n: number;
  source_name: string;
  page_or_section: string;
  score: number;
  doc_id: string;
  chunk_id: string;
  excerpt: string;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function req<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const res = await fetch(path, {
    credentials: "include",
    headers: opts.body ? { "Content-Type": "application/json" } : {},
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  const ct = res.headers.get("content-type") || "";
  return (ct.includes("application/json") ? await res.json() : (await res.text())) as T;
}

export const api = {
  // Auth
  login: (username: string, password: string) =>
    req<User>("/auth/login", { method: "POST", body: JSON.stringify({ username, password }) }),
  logout: () => req<void>("/auth/logout", { method: "POST" }),
  me: () => req<User>("/auth/me"),
  createUser: (username: string, password: string, role: string) =>
    req<User>("/auth/users", { method: "POST", body: JSON.stringify({ username, password, role }) }),

  // Collections
  listCollections: () => req<Collection[]>("/collections"),
  getCollection: (name: string) => req<Collection>(`/collections/${name}`),
  createCollection: (body: Record<string, unknown>) =>
    req<Collection>("/collections", { method: "POST", body: JSON.stringify(body) }),
  updateCollection: (name: string, body: Record<string, unknown>) =>
    req<Collection>(`/collections/${name}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteCollection: (name: string) => req<void>(`/collections/${name}`, { method: "DELETE" }),

  // Documents & jobs
  listDocuments: (name: string) => req<DocumentItem[]>(`/collections/${name}/documents`),
  deleteDocument: (name: string, docId: string) =>
    req<{ chunks_removed: number }>(`/collections/${name}/documents/${docId}`, { method: "DELETE" }),
  listJobs: (name: string) => req<Job[]>(`/collections/${name}/jobs`),
  getJob: (id: number) => req<Job>(`/jobs/${id}`),

  uploadDocuments: async (name: string, files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    const res = await fetch(`/collections/${name}/documents`, {
      method: "POST",
      credentials: "include",
      body: fd,
    });
    if (!res.ok) throw new ApiError(res.status, (await res.json().catch(() => ({}))).detail || "upload");
    return (await res.json()) as { jobs: { job_id: number; source_name: string; doc_id: string }[] };
  },

  feedback: (name: string, body: Record<string, unknown>) =>
    req<void>(`/collections/${name}/feedback`, { method: "POST", body: JSON.stringify(body) }),

  imageUrl: (name: string, ref: string) => {
    // ref = "rag-image://<collection>/<doc>/<file>" → /collections/<name>/images/<doc>/<file>
    const rel = ref.replace("rag-image://", "");
    const parts = rel.split("/");
    return `/collections/${name}/images/${parts.slice(1).join("/")}`;
  },
};

export interface StreamHandlers {
  onSources: (sources: Source[]) => void;
  onToken: (t: string) => void;
  onDone: (timings: Record<string, number>) => void;
  onError: (message: string) => void;
}

// Consomme le flux SSE de POST /collections/{name}/query.
export async function streamQuery(
  name: string,
  question: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`/collections/${name}/query`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!res.ok || !res.body) {
    handlers.onError(`HTTP ${res.status}`);
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const blocks = buf.split("\n\n");
    buf = blocks.pop() ?? "";
    for (const block of blocks) {
      if (!block.trim()) continue;
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;
      const parsed = JSON.parse(data);
      if (event === "sources") handlers.onSources(parsed.sources);
      else if (event === "token") handlers.onToken(parsed.t);
      else if (event === "done") handlers.onDone(parsed.timings);
      else if (event === "error") handlers.onError(parsed.message);
    }
  }
}
