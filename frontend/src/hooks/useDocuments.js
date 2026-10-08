import { useCallback, useState } from 'react'
import { deleteDocument, listDocuments, uploadDocument } from '../services/api.js'

/** Uploaded-document state for the sidebar (Phase 10 upload UI). */
export function useDocuments() {
  const [documents, setDocuments] = useState([])
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)

  const refresh = useCallback(async () => {
    try {
      setDocuments(await listDocuments())
    } catch {
      /* sidebar degrades quietly */
    }
  }, [])

  const upload = useCallback(async (file) => {
    setUploading(true)
    setError(null)
    try {
      await uploadDocument(file)
      setDocuments(await listDocuments())
    } catch (err) {
      setError(err.message ?? 'Upload failed.')
    } finally {
      setUploading(false)
    }
  }, [])

  const remove = useCallback(async (documentId) => {
    setError(null)
    try {
      await deleteDocument(documentId)
      setDocuments((prev) => prev.filter((doc) => doc.id !== documentId))
    } catch (err) {
      setError(err.message ?? 'Could not delete the document.')
    }
  }, [])

  return { documents, refresh, upload, remove, uploading, error, dismissError: () => setError(null) }
}
