import type { Device, EventStatus, FallEvent, PiState } from './types'

/** Thiet bi coi la online neu heartbeat gan nhat <= 30 giay (agent gui moi 10 giay) */
export const ONLINE_WINDOW_MS = 30_000

export function isOnline(d: Device, now = Date.now()): boolean {
  return !!d.last_seen && now - new Date(d.last_seen).getTime() <= ONLINE_WINDOW_MS
}

export type DeviceView = 'fall' | 'warn' | 'normal' | 'off' | 'nodetector'

export function deviceView(d: Device, now = Date.now()): DeviceView {
  if (!isOnline(d, now)) return 'off'
  if (d.state === 'fall') return 'fall'
  if (d.state === 'sit' || d.state === 'lie') return 'warn'
  if (d.state === 'unknown') return 'nodetector'
  return 'normal'
}

export const STATE_LABEL: Record<PiState, string> = {
  normal: 'Bình thường', sit: 'Ngồi', lie: 'Nằm', fall: 'Té ngã', warmup: 'Khởi động', unknown: 'Không rõ',
}

export const VIEW_LABEL: Record<DeviceView, string> = {
  fall: 'Té ngã', warn: 'Cảnh báo', normal: 'Bình thường', off: 'Offline', nodetector: 'Chưa chạy nhận diện',
}

export const VIEW_PILL: Record<DeviceView, string> = {
  fall: 'fall', warn: 'warn', normal: 'normal', off: 'off', nodetector: 'off',
}

export const STATUS_LABEL: Record<EventStatus, string> = {
  new: 'Mới', acknowledged: 'Đã xác nhận', resolved: 'Đã xử lý',
}
export const STATUS_PILL: Record<EventStatus, string> = { new: 'new', acknowledged: 'ack', resolved: 'done' }

export function deviceName(d: Device | undefined, fallbackId?: string): string {
  return d?.name || d?.id || fallbackId || 'Thiết bị'
}

const pad = (n: number) => String(n).padStart(2, '0')

export function fmtTime(iso: string | null | undefined, withSeconds = true): string {
  if (!iso) return '–'
  const d = new Date(iso)
  return `${pad(d.getHours())}:${pad(d.getMinutes())}${withSeconds ? ':' + pad(d.getSeconds()) : ''}`
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '–'
  const d = new Date(iso)
  return `${pad(d.getDate())}/${pad(d.getMonth() + 1)}/${d.getFullYear()}`
}

export function isSameDay(a: Date, b: Date): boolean {
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate()
}

export function fmtDayLabel(iso: string | null | undefined, now = new Date()): string {
  if (!iso) return '–'
  const d = new Date(iso)
  if (isSameDay(d, now)) return 'Hôm nay'
  const y = new Date(now); y.setDate(y.getDate() - 1)
  if (isSameDay(d, y)) return 'Hôm qua'
  return fmtDate(iso)
}

export function fmtDuration(sec: number | null | undefined): string {
  if (sec == null || !isFinite(sec)) return '–'
  if (sec < 60) return `${sec.toFixed(1).replace('.', ',')} giây`
  const m = Math.floor(sec / 60), s = Math.round(sec % 60)
  if (m < 60) return s ? `${m} phút ${s} giây` : `${m} phút`
  const h = Math.floor(m / 60)
  return `${h} giờ ${m % 60} phút`
}

export function fmtAgo(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return 'chưa có dữ liệu'
  const s = Math.max(0, Math.round((now - new Date(iso).getTime()) / 1000))
  if (s < 5) return 'vừa xong'
  if (s < 60) return `${s} giây trước`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m} phút trước`
  const h = Math.floor(m / 60)
  if (h < 24) return `${h} giờ trước`
  return `${Math.floor(h / 24)} ngày trước`
}

/** Thoi gian tu luc phat canh bao den khi co nguoi xac nhan (giay) */
export function responseSeconds(e: FallEvent): number | null {
  if (!e.acknowledged_at) return null
  return (new Date(e.acknowledged_at).getTime() - new Date(e.detected_at).getTime()) / 1000
}

export function fmtNumber(n: number, digits = 1): string {
  return n.toFixed(digits).replace('.', ',')
}

export function shortId(id: string): string {
  return '#' + id.slice(0, 8)
}
