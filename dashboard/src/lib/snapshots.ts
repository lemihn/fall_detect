import { useEffect, useState } from 'react'
import { supabase } from './supabase'

const cache = new Map<string, { url: string; exp: number }>()
const pending = new Map<string, Promise<string | null>>()
const TTL_S = 3600

/** Anh trong bucket private -> signed URL (luu cache 50 phut) */
export async function signedSnapshotUrl(path: string): Promise<string | null> {
  const hit = cache.get(path)
  if (hit && hit.exp > Date.now()) return hit.url
  if (pending.has(path)) return pending.get(path)!
  const p = supabase.storage.from('snapshots').createSignedUrl(path, TTL_S).then(({ data }) => {
    pending.delete(path)
    if (!data?.signedUrl) return null
    cache.set(path, { url: data.signedUrl, exp: Date.now() + (TTL_S - 600) * 1000 })
    return data.signedUrl
  })
  pending.set(path, p)
  return p
}

export function useSnapshot(path: string | null | undefined): string | null {
  const [url, setUrl] = useState<string | null>(path ? cache.get(path)?.url ?? null : null)
  useEffect(() => {
    if (!path) { setUrl(null); return }
    let alive = true
    signedSnapshotUrl(path).then(u => alive && setUrl(u))
    return () => { alive = false }
  }, [path])
  return url
}
