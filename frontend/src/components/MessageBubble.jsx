import { FileText, Sparkles } from 'lucide-react'
import MarkdownContent from './MarkdownContent.jsx'

function StreamingDots() {
  return (
    <div className="flex items-center gap-1.5 py-1" aria-label="NOVA is typing">
      <span className="dot h-2 w-2 rounded-full bg-slate-400" />
      <span className="dot h-2 w-2 rounded-full bg-slate-400" />
      <span className="dot h-2 w-2 rounded-full bg-slate-400" />
    </div>
  )
}

function SourceChips({ sources }) {
  if (!sources?.length) return null
  return (
    <div className="mt-2.5 flex flex-wrap gap-1.5">
      {sources.map((source, index) => (
        <span
          key={`${source.document_id ?? source.filename ?? 'src'}-${index}`}
          title={`${source.filename ?? source.document_id ?? 'document'} · chunk ${source.part} · score ${source.score?.toFixed(3)}`}
          className="inline-flex max-w-full items-center gap-1 rounded-md border border-slate-700 bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-400"
        >
          <FileText className="h-3 w-3 shrink-0 text-indigo-400" aria-hidden="true" />
          <span className="truncate">{source.filename ?? source.document_id ?? 'document'}</span>
          <span className="text-slate-500">
            p{source.part} · {source.score?.toFixed(2)}
          </span>
        </span>
      ))}
    </div>
  )
}

export default function MessageBubble({ message }) {
  const isUser = message.role === 'user'

  if (isUser) {
    return (
      <div className="animate-fade-up flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-md bg-indigo-600 px-4 py-2.5 text-sm leading-relaxed text-white shadow-md shadow-indigo-950/40 sm:max-w-[70%]">
          <p className="whitespace-pre-wrap break-words">{message.content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="animate-fade-up flex items-start gap-3">
      <div className="mt-1 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-slate-700 bg-slate-900">
        <Sparkles className="h-3.5 w-3.5 text-indigo-400" aria-hidden="true" />
      </div>
      <div className="max-w-[85%] rounded-2xl rounded-bl-md border border-slate-800 bg-slate-900 px-4 py-2.5 text-sm leading-relaxed text-slate-200 sm:max-w-[75%]">
        {message.streaming && !message.content ? (
          <StreamingDots />
        ) : (
          <>
            <MarkdownContent content={message.content} />
            {message.streaming && (
              <span
                className="ml-0.5 inline-block h-3.5 w-1.5 translate-y-0.5 animate-pulse bg-indigo-500"
                aria-hidden="true"
              />
            )}
          </>
        )}

        <SourceChips sources={message.sources} />

        {message.intent &&
          message.intent !== 'unknown' &&
          message.confidence != null && (
            <span className="mt-2 inline-block rounded-md bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-400">
              intent: {message.intent} · {(message.confidence * 100).toFixed(0)}%
            </span>
          )}
      </div>
    </div>
  )
}
