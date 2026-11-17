import { useEffect, useState } from 'react'
import { supabase } from './supabase'
import type { Device, PiStatus } from './types'
import { isOnline } from './format'

/** Lay access token hien tai (supabase-js tu lam moi truoc khi het han) */
export async function accessToken(): Promise<string | null> {
  const { data } = await supabase.auth.getSession()
  return data.session?.access_token ?? null
}

export function hasStream(d: Device | undefined, now = Date.now()): boolean {
  return !!d && !!d.stream_url && isOnline(d, now)
}

/** URL video MJPEG kem token, doi khi token duoc lam moi */
export function useStreamUrl(d: Device | undefined, now: number): string | null {
  const [token, setToken] = useState<string | null>(null)
  useEffect(() => {
    accessToken().then(setToken)
    const { data } = supabase.auth.onAuthStateChange((_e, s) => setToken(s?.access_token ?? null))
    return () => data.subscription.unsubscribe()
  }, [])
  if (!hasStream(d, now) || !token) return null
  return `${d!.stream_url}/stream?token=${encodeURIComponent(token)}`
}

async function call(d: Device, path: string, init?: RequestInit): Promise<Response> {
  const token = await accessToken()
  if (!d.stream_url || !token) throw new Error('Thiết bị chưa có kết nối camera')
  return fetch(d.stream_url + path, {
    ...init,
    headers: { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', ...(init?.headers || {}) },
  })
}

export async function resetAlarm(d: Device): Promise<string | null> {
  try {
    const r = await call(d, '/api/reset_alert', { method: 'POST' })
    return r.ok ? null : `Lỗi ${r.status}`
  } catch (e) {
    return (e as Error).message
  }
}

export async function switchCamera(d: Device, camera: number): Promise<string | null> {
  try {
    const r = await call(d, '/api/switch_camera', { method: 'POST', body: JSON.stringify({ camera }) })
    return r.ok ? null : `Lỗi ${r.status}`
  } catch (e) {
    return (e as Error).message
  }
}

/** Doc /status cua Pi qua gateway moi giay (xac suat 4 tu the, nguoi dang theo doi) */
export function usePiStatus(d: Device | undefined, now: number, enabled = true): PiStatus | null {
  const [st, setSt] = useState<PiStatus | null>(null)
  const ok = enabled && hasStream(d, now)
  const url = d?.stream_url
  useEffect(() => {
    if (!ok || !d) { setSt(null); return }
    let alive = true
    let timer: ReturnType<typeof setTimeout>
    const tick = async () => {
      try {
        const r = await call(d, '/api/status')
        if (r.ok && alive) setSt(await r.json())
      } catch { /* bo qua, lan sau thu lai */ }
      if (alive) timer = setTimeout(tick, 1000)
    }
    tick()
    return () => { alive = false; clearTimeout(timer) }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ok, url, d?.id])
  return st
}
