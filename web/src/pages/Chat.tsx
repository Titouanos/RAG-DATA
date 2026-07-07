import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router-dom";

import { api, streamQuery, type Source } from "../api";
import { CollectionTabs } from "../ui";

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  timings?: Record<string, number>;
  error?: string;
  streaming?: boolean;
  feedback?: "up" | "down";
}

export default function Chat() {
  const { name = "" } = useParams();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const send = async () => {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setBusy(true);
    const question = q;
    setMessages((m) => [
      ...m,
      { role: "user", content: question },
      { role: "assistant", content: "", streaming: true },
    ]);

    const patchLast = (patch: Partial<Message>) =>
      setMessages((m) => {
        const copy = [...m];
        copy[copy.length - 1] = { ...copy[copy.length - 1], ...patch };
        return copy;
      });
    const appendToken = (t: string) =>
      setMessages((m) => {
        const copy = [...m];
        const last = copy[copy.length - 1];
        copy[copy.length - 1] = { ...last, content: last.content + t };
        return copy;
      });

    try {
      await streamQuery(name, question, {
        onSources: (sources) => patchLast({ sources }),
        onToken: appendToken,
        onDone: (timings) => patchLast({ timings, streaming: false }),
        onError: (message) => patchLast({ error: message, streaming: false }),
      });
    } catch (e) {
      patchLast({ error: String(e), streaming: false });
    } finally {
      patchLast({ streaming: false });
      setBusy(false);
    }
  };

  return (
    <div>
      <h1 className="mb-4 text-xl font-semibold">{name}</h1>
      <CollectionTabs name={name} />

      <div className="space-y-4">
        {messages.length === 0 && (
          <div className="card p-8 text-center text-slate-400">
            Posez une question sur les documents de « {name} ».
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "user" ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-[80%] rounded-lg bg-brand-600 px-4 py-2 text-sm text-white">
                {m.content}
              </div>
            </div>
          ) : (
            <AssistantMessage key={i} msg={m} collection={name} question={messages[i - 1]?.content ?? ""} />
          ),
        )}
        <div ref={bottom} />
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
        className="sticky bottom-4 mt-4 flex gap-2"
      >
        <input
          className="input flex-1 shadow-sm"
          placeholder="Votre question…"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          disabled={busy}
        />
        <button className="btn-primary" disabled={busy || !input.trim()}>
          {busy ? "…" : "Envoyer"}
        </button>
      </form>
    </div>
  );
}

function AssistantMessage({
  msg,
  collection,
  question,
}: {
  msg: Message;
  collection: string;
  question: string;
}) {
  const [fb, setFb] = useState<"up" | "down" | undefined>(msg.feedback);

  const sendFeedback = async (rating: "up" | "down") => {
    setFb(rating);
    try {
      await api.feedback(collection, {
        question,
        rating,
        answer_excerpt: msg.content.slice(0, 500),
        chunk_ids: (msg.sources ?? []).map((s) => s.chunk_id),
      });
    } catch {
      /* silencieux */
    }
  };

  return (
    <div className="card p-4">
      <div className="prose-sm max-w-none text-sm text-slate-800">
        {msg.content ? (
          <RichText text={msg.content} collection={collection} />
        ) : msg.streaming ? (
          <span className="text-slate-400">Recherche et rédaction…</span>
        ) : null}
        {msg.streaming && msg.content && <span className="ml-0.5 animate-pulse">▍</span>}
      </div>

      {msg.error && (
        <div className="mt-2 rounded bg-red-50 px-3 py-2 text-sm text-red-700">
          Erreur : {msg.error}
        </div>
      )}

      {msg.sources && msg.sources.length > 0 && (
        <div className="mt-3 border-t border-slate-100 pt-3">
          <div className="mb-1 text-xs font-semibold uppercase text-slate-400">Sources</div>
          <div className="space-y-1">
            {msg.sources.map((s) => (
              <SourceCard key={s.n} s={s} />
            ))}
          </div>
        </div>
      )}

      {!msg.streaming && msg.content && (
        <div className="mt-3 flex items-center gap-2 border-t border-slate-100 pt-2 text-xs text-slate-400">
          <button
            className={`btn-ghost px-2 py-1 ${fb === "up" ? "text-green-600" : ""}`}
            onClick={() => sendFeedback("up")}
          >
            👍
          </button>
          <button
            className={`btn-ghost px-2 py-1 ${fb === "down" ? "text-red-600" : ""}`}
            onClick={() => sendFeedback("down")}
          >
            👎
          </button>
          {msg.timings && (
            <span className="ml-auto">
              {Math.round(msg.timings.total_ms)} ms (embed {Math.round(msg.timings.embed_ms)} · search{" "}
              {Math.round(msg.timings.search_ms)} · rerank {Math.round(msg.timings.rerank_ms)})
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function SourceCard({ s }: { s: Source }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded border border-slate-200">
      <button
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm hover:bg-slate-50"
        onClick={() => setOpen((o) => !o)}
      >
        <span className="badge bg-brand-50 text-brand-700">[{s.n}]</span>
        <span className="font-medium text-slate-700">{s.source_name}</span>
        <span className="text-xs text-slate-400">{s.page_or_section}</span>
        <span className="ml-auto text-xs text-slate-400">score {s.score.toFixed(3)}</span>
      </button>
      {open && (
        <div className="whitespace-pre-wrap border-t border-slate-100 px-3 py-2 text-xs text-slate-600">
          {s.excerpt}
        </div>
      )}
    </div>
  );
}

// Rendu léger : remplace les images ![alt](rag-image://…) par des <img>, texte en pré-wrap.
function RichText({ text, collection }: { text: string; collection: string }) {
  const re = /!\[([^\]]*)\]\((rag-image:\/\/[^)]+)\)/g;
  const parts: React.ReactNode[] = [];
  let last = 0;
  let m: RegExpExecArray | null;
  let key = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(<span key={key++}>{text.slice(last, m.index)}</span>);
    parts.push(
      <img
        key={key++}
        src={api.imageUrl(collection, m[2])}
        alt={m[1]}
        className="my-2 max-h-72 rounded border border-slate-200"
      />,
    );
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push(<span key={key++}>{text.slice(last)}</span>);
  return <div className="whitespace-pre-wrap">{parts}</div>;
}
