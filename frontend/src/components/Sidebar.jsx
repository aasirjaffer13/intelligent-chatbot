import { useRef } from 'react'
import {
  FileText,
  Loader2,
  MessageSquare,
  Plus,
  Sparkles,
  Trash2,
  Upload,
  X,
} from 'lucide-react'

/**
 * Conversation + document sidebar (Phase 10).
 *
 * - Conversations: click to reopen a past chat (replays stored history).
 * - Documents: upload PDF/txt/md for RAG, delete them again.
 * - Responsive: static panel on desktop, overlay with backdrop on mobile.
 */
export default function Sidebar({
  open,
  onClose,
  conversations,
  activeSessionId,
  onSelect,
  onNew,
  conversationsLoading,
  documents,
  uploading,
  documentError,
  onUpload,
  onDeleteDocument,
}) {
  const fileInputRef = useRef(null)

  function handleFileChange(event) {
    const file = event.target.files?.[0]
    if (file) onUpload(file)
    event.target.value = '' // allow re-selecting the same file
  }

  return (
    <>
      {open && (
        <button
          type="button"
          aria-label="Close sidebar"
          onClick={onClose}
          className="fixed inset-0 z-30 bg-black/50 md:hidden"
        />
      )}
      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-slate-800 bg-slate-950 transition-transform duration-200 md:static md:z-auto md:w-64 md:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* brand */}
        <div className="flex items-center justify-between border-b border-slate-800 px-4 py-3">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-indigo-500 to-violet-600 shadow-lg shadow-indigo-950/50">
              <Sparkles className="h-4 w-4 text-white" aria-hidden="true" />
            </div>
            <span className="text-sm font-semibold tracking-wide text-slate-100">NOVA</span>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close sidebar"
            className="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-800 hover:text-slate-200 md:hidden"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {/* new chat */}
        <div className="px-3 pt-3">
          <button
            type="button"
            onClick={onNew}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-900 px-3 py-2 text-sm font-medium text-slate-200 transition hover:border-indigo-500 hover:bg-slate-800 hover:text-white"
          >
            <Plus className="h-4 w-4 text-indigo-400" aria-hidden="true" />
            New chat
          </button>
        </div>

        {/* conversations */}
        <div className="scroll-slim mt-4 flex-1 overflow-y-auto px-3 pb-3">
          <p className="px-1 pb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Conversations
          </p>
          {conversationsLoading ? (
            <div className="space-y-2 px-1">
              {[0, 1, 2].map((i) => (
                <div key={i} className="h-9 animate-pulse rounded-lg bg-slate-900" />
              ))}
            </div>
          ) : conversations.length === 0 ? (
            <p className="px-1 text-xs leading-relaxed text-slate-500">
              No conversations yet. Say hi to start one.
            </p>
          ) : (
            <ul className="space-y-1">
              {conversations.map((conversation) => {
                const isActive = conversation.session_id === activeSessionId
                return (
                  <li key={conversation.session_id}>
                    <button
                      type="button"
                      onClick={() => onSelect(conversation.session_id)}
                      title={conversation.preview}
                      className={`w-full rounded-lg border px-2.5 py-2 text-left transition ${
                        isActive
                          ? 'border-indigo-500/70 bg-slate-900'
                          : 'border-transparent hover:border-slate-800 hover:bg-slate-900'
                      }`}
                    >
                      <span className="flex items-center gap-1.5">
                        <MessageSquare
                          className={`h-3.5 w-3.5 shrink-0 ${isActive ? 'text-indigo-400' : 'text-slate-500'}`}
                          aria-hidden="true"
                        />
                        <span className="truncate text-xs text-slate-300">
                          {conversation.preview}
                        </span>
                      </span>
                      <span className="mt-0.5 block pl-5 text-[10px] text-slate-500">
                        {conversation.message_count}{' '}
                        {conversation.message_count === 1 ? 'message' : 'messages'}
                      </span>
                    </button>
                  </li>
                )
              })}
            </ul>
          )}

          {/* documents */}
          <p className="px-1 pb-2 pt-5 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Documents
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md,text/plain,text/markdown,application/pdf"
            onChange={handleFileChange}
            className="hidden"
            aria-label="Upload a document"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="flex w-full items-center justify-center gap-2 rounded-xl border border-dashed border-slate-700 px-3 py-2 text-xs text-slate-400 transition hover:border-indigo-500 hover:text-indigo-300 disabled:cursor-wait disabled:opacity-60"
          >
            {uploading ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
                Uploading…
              </>
            ) : (
              <>
                <Upload className="h-3.5 w-3.5" aria-hidden="true" />
                Upload PDF / txt / md
              </>
            )}
          </button>

          {documentError && (
            <p className="mt-2 rounded-lg border border-rose-900/60 bg-rose-950/50 px-2 py-1.5 text-[11px] leading-relaxed text-rose-300">
              {documentError}
            </p>
          )}

          {documents.length === 0 ? (
            <p className="mt-2 px-1 text-xs leading-relaxed text-slate-500">
              Upload a document and ask NOVA questions about it.
            </p>
          ) : (
            <ul className="mt-2 space-y-1">
              {documents.map((document) => (
                <li
                  key={document.id}
                  className="group flex items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-900 px-2.5 py-2"
                >
                  <FileText className="h-3.5 w-3.5 shrink-0 text-indigo-400" aria-hidden="true" />
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-xs text-slate-300">{document.filename}</span>
                    <span className="block text-[10px] text-slate-500">
                      {document.chunk_count} {document.chunk_count === 1 ? 'chunk' : 'chunks'}
                    </span>
                  </span>
                  <button
                    type="button"
                    onClick={() => onDeleteDocument(document.id)}
                    aria-label={`Delete ${document.filename}`}
                    className="rounded p-1 text-slate-500 opacity-0 transition hover:bg-rose-950/60 hover:text-rose-400 focus-visible:opacity-100 group-hover:opacity-100"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* footer */}
        <div className="border-t border-slate-800 px-4 py-2.5">
          <p className="text-[10px] leading-relaxed text-slate-500">
            NOVA · Intelligent NLP Chatbot · Phase 10
          </p>
        </div>
      </aside>
    </>
  )
}
