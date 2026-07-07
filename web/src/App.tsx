import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";

import { useAuth } from "./auth";
import Chat from "./pages/Chat";
import CollectionDetail from "./pages/CollectionDetail";
import Collections from "./pages/Collections";
import Login from "./pages/Login";
import Settings from "./pages/Settings";

function Header() {
  const { user, logout } = useAuth();
  return (
    <header className="border-b border-slate-200 bg-white">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3">
        <Link to="/" className="text-lg font-semibold text-brand-700">
          RAG Builder
        </Link>
        {user && (
          <div className="flex items-center gap-3 text-sm">
            <span className="text-slate-500">
              {user.username}
              <span className="ml-1 badge bg-slate-100 text-slate-600">{user.role}</span>
            </span>
            <button className="btn-ghost" onClick={() => logout()}>
              Déconnexion
            </button>
          </div>
        )}
      </div>
    </header>
  );
}

export default function App() {
  const { user, loading } = useAuth();
  const location = useLocation();

  if (loading) {
    return <div className="p-10 text-center text-slate-400">Chargement…</div>;
  }
  if (!user) {
    return location.pathname === "/login" ? (
      <Login />
    ) : (
      <Navigate to="/login" replace state={{ from: location.pathname }} />
    );
  }

  return (
    <div className="min-h-screen">
      <Header />
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Routes>
          <Route path="/" element={<Collections />} />
          <Route path="/c/:name" element={<CollectionDetail />} />
          <Route path="/c/:name/chat" element={<Chat />} />
          <Route path="/c/:name/settings" element={<Settings />} />
          <Route path="/login" element={<Navigate to="/" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}
