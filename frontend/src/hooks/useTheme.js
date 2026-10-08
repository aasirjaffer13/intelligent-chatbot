import { useCallback, useEffect, useState } from 'react'

const STORAGE_KEY = 'nova-theme'

function readInitialTheme() {
  try {
    return localStorage.getItem(STORAGE_KEY) === 'light' ? 'light' : 'dark'
  } catch {
    return 'dark'
  }
}

/**
 * Dark/light theme backed by a `light` class on <html>.
 *
 * The whole palette flips because Tailwind v4 utilities compile to
 * `var(--color-*)` — index.css redefines those variables under
 * `:root.light`, so no component needs light-specific classes.
 */
export function useTheme() {
  const [theme, setTheme] = useState(readInitialTheme)

  useEffect(() => {
    document.documentElement.classList.toggle('light', theme === 'light')
    try {
      localStorage.setItem(STORAGE_KEY, theme)
    } catch {
      /* storage unavailable (private mode) — theme still applies */
    }
  }, [theme])

  const toggle = useCallback(() => {
    setTheme((current) => (current === 'dark' ? 'light' : 'dark'))
  }, [])

  return { theme, toggle }
}
