import { useRef, useState } from 'react'
import { ArrowUp } from 'lucide-react'

const MAX_LENGTH = 4000

export default function ChatInput({ onSend, disabled }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  const canSend = value.trim().length > 0 && !disabled

  function resize() {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`
  }

  function submit() {
    if (!canSend) return
    onSend(value)
    setValue('')
    requestAnimationFrame(() => {
      if (textareaRef.current) textareaRef.current.style.height = 'auto'
    })
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <div className="border-t border-slate-800 bg-slate-950 px-4 pb-4 pt-3 sm:px-6">
      <div className="mx-auto w-full max-w-3xl">
        <div className="flex items-end gap-2 rounded-2xl border border-slate-700 bg-slate-900 p-2 shadow-lg transition focus-within:border-indigo-500">
          <textarea
            ref={textareaRef}
            rows={1}
            value={value}
            maxLength={MAX_LENGTH}
            onChange={(event) => {
              setValue(event.target.value)
              resize()
            }}
            onKeyDown={handleKeyDown}
            placeholder="Message NOVA…"
            aria-label="Message NOVA"
            className="max-h-40 flex-1 resize-none bg-transparent px-2 py-1.5 text-sm text-slate-100 placeholder-slate-500 outline-none"
          />
          <button
            type="button"
            onClick={submit}
            disabled={!canSend}
            aria-label="Send message"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-indigo-600 text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {disabled ? (
              <span className="h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
            ) : (
              <ArrowUp className="h-4 w-4" />
            )}
          </button>
        </div>
        <p className="mt-2 text-center text-[11px] text-slate-500">
          Enter to send · Shift + Enter for a new line
          {value.length > 0 && ` · ${value.length}/${MAX_LENGTH}`}
        </p>
      </div>
    </div>
  )
}
