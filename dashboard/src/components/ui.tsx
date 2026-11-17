import { createContext, useCallback, useContext, useEffect, useState, type ReactNode } from 'react'
import { Icon } from './Icon'
import { useSnapshot } from '../lib/snapshots'
import { streamBase, useStreamUrl } from '../lib/gateway'
import type { Device, EventStatus } from '../lib/types'
import { STATUS_LABEL, STATUS_PILL, VIEW_LABEL, VIEW_PILL, deviceView, fmtAgo, isOnline } from '../lib/format'

export function Pill({ kind, children, plain }: { kind: string; children: ReactNode; plain?: boolean }) {
  return <span className={`pill ${kind}${plain ? ' plain' : ''}`}>{children}</span>
}

export function DevicePill({ d, now }: { d: Device; now: number }) {
  const v = deviceView(d, now)
  return <Pill kind={VIEW_PILL[v]}>{VIEW_LABEL[v]}</Pill>
}

export function StatusPill({ s }: { s: EventStatus }) {
  return <Pill kind={STATUS_PILL[s]}>{STATUS_LABEL[s]}</Pill>
}

export function Esp32Pill({ d, now }: { d: Device; now: number }) {
  if (!isOnline(d, now) || !d.esp32_online) return <Pill kind="off" plain>Không rõ</Pill>
  if (d.esp32_state === 'fall') return <Pill kind="fall" plain>Đang hú còi</Pill>
  if (d.esp32_state === 'warning') return <Pill kind="warn" plain>Cảnh báo</Pill>
  return <Pill kind="normal" plain>Online</Pill>
}

export function Thumb({ path }: { path: string | null }) {
  const url = useSnapshot(path)
  return <span className="thumb">{url ? <img src={url} alt="Ảnh hiện trường" loading="lazy" /> : <Icon name="img" />}</span>
}

export function Snapshot({ path, label }: { path: string | null; label?: string }) {
  const url = useSnapshot(path)
  return (
    <div className="shot">
      {url ? <img src={url} alt="Ảnh hiện trường lúc cảnh báo" /> : <div className="noimg">{path ? 'Đang tải ảnh…' : 'Không có ảnh'}</div>}
      {label && url && <span className="tag live" style={{ position: 'absolute', top: 10, left: 10 }}>{label}</span>}
    </div>
  )
}

/** Video live tu gateway tren Pi; tu hien thong bao khi offline / loi */
export function StreamView({ d, now, overlay }: { d: Device | undefined; now: number; overlay?: ReactNode }) {
  const src = useStreamUrl(d, now)
  const [failed, setFailed] = useState(false)
  const [retry, setRetry] = useState(0)
  useEffect(() => { setFailed(false) }, [src])
  useEffect(() => {
    if (!failed) return
    const t = setTimeout(() => { setFailed(false); setRetry(r => r + 1) }, 5000)
    return () => clearTimeout(t)
  }, [failed])

  let msg: ReactNode = null
  if (!d) msg = 'Chưa chọn thiết bị'
  else if (!isOnline(d, now)) msg = <><Icon name="wifioff" />Camera offline<br /><span style={{ fontSize: 12 }}>Mất kết nối {fmtAgo(d.last_seen, now)}</span></>
  else if (!d.stream_url) msg = <><Icon name="cam" />Chưa có kết nối camera từ xa<br /><span style={{ fontSize: 12 }}>Kiểm tra cloudflared hoặc PUBLIC_STREAM_URL trên Pi</span></>
  else if (!streamBase(d)) msg = <><Icon name="warn" />Link camera không hợp lệ<br /><span style={{ fontSize: 12 }}>Kiểm tra PUBLIC_STREAM_URL trong dashboard_agent/.env</span></>
  else if (failed) msg = <><Icon name="refresh" />Không tải được video<br /><span style={{ fontSize: 12 }}>Đang thử lại…</span></>
  else if (!src) msg = 'Đang kết nối…'

  return (
    <div className="cam">
      {msg ? <div className="placeholder"><div>{msg}</div></div> : (
        <img key={retry} className="stream" src={src!} alt={`Camera ${d?.name}`} onError={() => setFailed(true)} />
      )}
      {!msg && <span className="tag live">TRỰC TIẾP</span>}
      {!msg && overlay}
    </div>
  )
}

export function Empty({ icon = 'inbox', children }: { icon?: string; children: ReactNode }) {
  return <div className="empty"><Icon name={icon} />{children}</div>
}

/* ---------------- toast ---------------- */
type Toast = { text: string; error?: boolean } | null
const ToastCtx = createContext<(text: string, error?: boolean) => void>(() => {})

export function ToastProvider({ children }: { children: ReactNode }) {
  const [t, setT] = useState<Toast>(null)
  useEffect(() => {
    if (!t) return
    const id = setTimeout(() => setT(null), 3500)
    return () => clearTimeout(id)
  }, [t])
  const show = useCallback((text: string, error?: boolean) => setT({ text, error }), [])
  return (
    <ToastCtx.Provider value={show}>
      {children}
      {t && <div className={`toast${t.error ? ' error' : ''}`} role="status">{t.text}</div>}
    </ToastCtx.Provider>
  )
}

export const useToast = () => useContext(ToastCtx)
