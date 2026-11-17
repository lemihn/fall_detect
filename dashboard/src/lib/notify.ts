import type { FallEvent } from './types'
import { fmtDuration } from './format'

const KEY_SOUND = 'fd.sound'
const KEY_NOTIFY = 'fd.notify'

export const prefs = {
  get sound(): boolean { return localStorage.getItem(KEY_SOUND) !== '0' },
  set sound(v: boolean) { localStorage.setItem(KEY_SOUND, v ? '1' : '0') },
  get notify(): boolean { return localStorage.getItem(KEY_NOTIFY) === '1' },
  set notify(v: boolean) { localStorage.setItem(KEY_NOTIFY, v ? '1' : '0') },
}

let audio: AudioContext | null = null

/** Chuong canh bao 2 tong, tao bang WebAudio (khong can file am thanh) */
export function playAlarm(): void {
  try {
    audio = audio || new AudioContext()
    const ctx = audio
    const t0 = ctx.currentTime
    for (let i = 0; i < 3; i++) {
      for (const [f, dt] of [[880, 0], [660, 0.22]] as const) {
        const o = ctx.createOscillator(), g = ctx.createGain()
        o.type = 'square'
        o.frequency.value = f
        const t = t0 + i * 0.5 + dt
        g.gain.setValueAtTime(0.0001, t)
        g.gain.exponentialRampToValueAtTime(0.12, t + 0.02)
        g.gain.exponentialRampToValueAtTime(0.0001, t + 0.2)
        o.connect(g).connect(ctx.destination)
        o.start(t)
        o.stop(t + 0.21)
      }
    }
  } catch {
    /* trinh duyet chan am thanh khi chua tuong tac - bo qua */
  }
}

export async function requestNotifyPermission(): Promise<boolean> {
  if (!('Notification' in window)) return false
  if (Notification.permission === 'granted') return true
  if (Notification.permission === 'denied') return false
  return (await Notification.requestPermission()) === 'granted'
}

export function notifyNewFall(device: string, e: FallEvent): void {
  if (prefs.sound) playAlarm()
  if (prefs.notify && 'Notification' in window && Notification.permission === 'granted') {
    try {
      const n = new Notification('PHÁT HIỆN TÉ NGÃ · ' + device, {
        body: `Người #${e.person_id ?? '?'} nằm bất động ${fmtDuration(e.duration_s)}. Bấm để xem.`,
        tag: e.id,
        requireInteraction: true,
      })
      n.onclick = () => { window.focus(); window.location.assign('/alerts?id=' + e.id) }
    } catch {
      /* mot so trinh duyet mobile khong ho tro new Notification */
    }
  }
}
