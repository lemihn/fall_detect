export type PiState = 'normal' | 'sit' | 'lie' | 'fall' | 'warmup' | 'unknown'
export type EventStatus = 'new' | 'acknowledged' | 'resolved'
export type Role = 'admin' | 'viewer'

export interface Device {
  id: string
  name: string
  location: string
  state: PiState
  alarm: number
  fps: number | null
  camera: string | null
  persons_count: number
  last_seen: string | null
  stream_url: string | null
  stream_updated_at: string | null
  esp32_online: boolean
  esp32_state: 'normal' | 'warning' | 'fall' | null
  esp32_last_seen: string | null
  esp32_battery: number | null
  agent_version: string | null
  created_at: string
  updated_at: string
}

export interface FallEvent {
  id: string
  device_id: string
  person_id: number | null
  type: 'fall'
  fall_time: string | null
  detected_at: string
  duration_s: number | null
  snapshot_path: string | null
  status: EventStatus
  acknowledged_at: string | null
  acknowledged_by: string | null
  resolved_at: string | null
  resolved_by: string | null
  note: string
  created_at: string
}

export interface DashboardUser {
  user_id: string
  email: string
  full_name: string
  role: Role
}

/** /status cua 4_realtime.py, lay qua gateway /api/status */
export interface PiStatus {
  state: PiState
  alarm: number
  since: number
  fps: number
  probs: number[]
  camera: string
  persons: { id: number; state: PiState; since: number; fall_time: string | null; telegram_sent: boolean }[]
}
