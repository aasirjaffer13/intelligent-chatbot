import { AlertCircle, X } from 'lucide-react'
import ChatHeader from '../components/ChatHeader.jsx'
import MessageList from '../components/MessageList.jsx'
import ChatInput from '../components/ChatInput.jsx'
import { useChat, useHealth } from '../hooks/useChat.js'

export default function ChatPage() {
  const { messages, isSending, error, send, clearChat, dismissError } = useChat()
  const healthStatus = useHealth()

  return (
    <div className="flex h-full flex-col bg-slate-950">
      <ChatHeader status={healthStatus} onClear={clearChat} />

      <main className="scroll-slim flex-1 overflow-y-auto bg-gradient-to-b from-slate-950 via-slate-950 to-slate-900/50">
        <MessageList messages={messages} isSending={isSending} onSuggest={send} />
      </main>

      {error && (
        <div className="px-4 pt-3 sm:px-6">
          <div className="mx-auto flex w-full max-w-3xl items-start gap-2 rounded-xl border border-rose-900/60 bg-rose-950/50 px-4 py-3 text-sm text-rose-200">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            <p className="flex-1">{error}</p>
            <button
              type="button"
              onClick={dismissError}
              aria-label="Dismiss error"
              className="rounded p-1 text-rose-300 transition hover:bg-rose-900/50"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}

      <ChatInput onSend={send} disabled={isSending} />
    </div>
  )
}
