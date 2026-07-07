import { NavLink } from "react-router-dom";

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-slate-400">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-300 border-t-brand-500" />
      {label}
    </div>
  );
}

const STATUS_STYLE: Record<string, string> = {
  indexed: "bg-green-100 text-green-700",
  succeeded: "bg-green-100 text-green-700",
  pending: "bg-amber-100 text-amber-700",
  running: "bg-blue-100 text-blue-700",
  processing: "bg-blue-100 text-blue-700",
  skipped: "bg-slate-100 text-slate-600",
  failed: "bg-red-100 text-red-700",
};

export function StatusBadge({ status }: { status: string }) {
  return <span className={`badge ${STATUS_STYLE[status] ?? "bg-slate-100 text-slate-600"}`}>{status}</span>;
}

export function CollectionTabs({ name }: { name: string }) {
  const tabs = [
    { to: `/c/${name}`, label: "Documents", end: true },
    { to: `/c/${name}/chat`, label: "Chat", end: false },
    { to: `/c/${name}/settings`, label: "Paramètres", end: false },
  ];
  return (
    <nav className="mb-4 flex gap-1 border-b border-slate-200">
      {tabs.map((t) => (
        <NavLink
          key={t.to}
          to={t.to}
          end={t.end}
          className={({ isActive }) =>
            `px-3 py-2 text-sm font-medium border-b-2 -mb-px ${
              isActive ? "border-brand-600 text-brand-700" : "border-transparent text-slate-500 hover:text-slate-700"
            }`
          }
        >
          {t.label}
        </NavLink>
      ))}
    </nav>
  );
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} o`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} Ko`;
  return `${(n / 1024 / 1024).toFixed(1)} Mo`;
}

export function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short" });
  } catch {
    return iso;
  }
}
