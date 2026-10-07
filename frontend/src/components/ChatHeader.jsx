import { Sparkles, Trash2 } from 'lucide-react'

const STATUS_STYLES = {
  online: { dot: 'bg-emerald-400', text: 'text-emerald-300', label: 'Connected' },
  offline: { dot: 'bg-rose-500', text: 'text-rose-300', label: 'Offline' },
  checking: { dot: 'bg-amber-400 animate-pulse', text: 'text-amber-300', label: 'Connecting' },
}

export default function ChatHeader({ status, onClear }) {
  const style = STATUS_STYLES[status] ?? STATUS_STYLES.checking

  return (
    <header className="flex items-center justify-between border-b border-slate-800 bg-slate-950/80 px-4 py-3 backdrop-blur sm:px-6">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 shadow-lg shadow-indigo-950/50">
          <Sparkles className="h-5 w-5 text-white" aria-hidden="true" />
        </div>
        <div>
          <h1 className="text-sm font-semibold tracking-wide text-slate-100">NOVA</h1>
          <p className="text-xs text-slate-400">Intelligent NLP Chatbot</p>
        </div>
        <span className="ml-2 hidden rounded-full border border-slate-700 bg-slate-900 px-2.5 py-0.5 text-[11px] font-medium text-slate-400 sm:inline">
          Phase 1 · Foundation
        </span>
      </div>

      <div className="flex items-center gap-2 sm:gap-3">
        <span
          className={`flex items-center gap-1.5 rounded-full border border-slate-800 bg-slate-900 px-2.5 py-1 text-xs ${style.text}`}
          title={`Backend health: ${style.label}`}
        >
          <span className={`h-1.5 w-1.5 rounded-full ${style.dot}`} />
          {style.label}
        </span>
        <button
          type="button"
          onClick={onClear}
          className="rounded-lg p-2 text-slate-400 transition hover:bg-slate-800 hover:text-slate-200"
          title="Clear conversation"
          aria-label="Clear conversation"
        >
          <Trash2 className="h-4 w-4" />
        </button>
      </div>
    </header>
  )
}
