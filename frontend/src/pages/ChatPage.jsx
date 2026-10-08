import { useEffect, useState } from 'react'
import { AlertCircle, X } from 'lucide-react'
import ChatHeader from '../components/ChatHeader.jsx'
import MessageList from '../components/MessageList.jsx'
import ChatInput from '../components/ChatInput.jsx'
import Sidebar from '../components/Sidebar.jsx'
import { useChat, useHealth } from '../hooks/useChat.js'
import { useDocuments } from '../hooks/useDocuments.js'
import { useTheme } from '../hooks/useTheme.js'
import { getStatus } from '../services/api.js'

export default function ChatPage() {
  const {
    messages,
    isSending,
    error,
    send,
    clearChat,
    dismissError,
    conversations,
    conversationsLoading,
    refreshConversations,
    activeSessionId,
    loadConversation,
    isLoadingHistory,
  } = useChat()
  const { documents, refresh: refreshDocuments, upload, remove, uploading, error: documentError } =
    useDocuments()
  const healthStatus = useHealth()
  const { theme, toggle: toggleTheme } = useTheme()
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [runtime, setRuntime] = useState(null)

  useEffect(() => {
    getStatus().then(setRuntime).catch(() => setRuntime(null))
    refreshConversations()
    refreshDocuments()
  }, [refreshConversations, refreshDocuments])

  // refresh the sidebar whenever a turn finishes
  useEffect(() => {
    if (!isSending) refreshConversations()
  }, [isSending, refreshConversations])

  function selectConversation(sessionId) {
    loadConversation(sessionId)
    setSidebarOpen(false)
  }

  function newConversation() {
    clearChat()
    setSidebarOpen(false)
  }

  return (
    <div className="flex h-full bg-slate-950">
      <Sidebar
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        conversations={conversations}
        conversationsLoading={conversationsLoading}
        activeSessionId={activeSessionId}
        onSelect={selectConversation}
        onNew={newConversation}
        documents={documents}
        uploading={uploading}
        documentError={documentError}
        onUpload={upload}
        onDeleteDocument={remove}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <ChatHeader
          status={healthStatus}
          onClear={clearChat}
          onMenu={() => setSidebarOpen(true)}
          theme={theme}
          onToggleTheme={toggleTheme}
          runtime={runtime}
        />

        <main className="scroll-slim flex-1 overflow-y-auto bg-gradient-to-b from-slate-950 via-slate-950 to-slate-900/50">
          <MessageList
            messages={messages}
            isSending={isSending}
            isLoadingHistory={isLoadingHistory}
            onSuggest={send}
          />
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
    </div>
  )
}
