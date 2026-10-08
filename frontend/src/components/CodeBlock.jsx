import { useState } from 'react'
import { Check, Copy } from 'lucide-react'

/** Fenced code block with a language label and one-click copy. */
export default function CodeBlock({ language, code }) {
  const [copied, setCopied] = useState(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      /* clipboard unavailable — button silently does nothing */
    }
  }

  return (
    <div className="my-2 overflow-hidden rounded-xl border border-slate-700 bg-slate-950">
      <div className="flex items-center justify-between border-b border-slate-800 bg-slate-900 px-3 py-1.5">
        <span className="font-mono text-[10px] uppercase tracking-wide text-slate-400">
          {language || 'code'}
        </span>
        <button
          type="button"
          onClick={copy}
          aria-label={copied ? 'Copied' : 'Copy code'}
          className="flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] text-slate-400 transition hover:bg-slate-800 hover:text-slate-200"
        >
          {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre className="scroll-slim overflow-x-auto p-3 text-xs leading-relaxed text-slate-200">
        <code>{code}</code>
      </pre>
    </div>
  )
}
