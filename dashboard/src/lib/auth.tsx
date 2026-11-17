import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import type { Session } from '@supabase/supabase-js'
import { supabase } from './supabase'
import type { DashboardUser } from './types'

interface AuthState {
  loading: boolean
  session: Session | null
  /** null khi da dang nhap nhung khong co trong dashboard_users */
  profile: DashboardUser | null
  signIn: (email: string, password: string) => Promise<string | null>
  signOut: () => Promise<void>
}

const Ctx = createContext<AuthState | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null)
  const [profile, setProfile] = useState<DashboardUser | null>(null)
  const [loading, setLoading] = useState(true)
  const [ready, setReady] = useState(false)

  useEffect(() => {
    let alive = true
    supabase.auth.getSession().then(({ data }) => {
      if (!alive) return
      setSession(data.session)
      setReady(true)
    })
    const { data: sub } = supabase.auth.onAuthStateChange((_ev, s) => setSession(s))
    return () => { alive = false; sub.subscription.unsubscribe() }
  }, [])

  const uid = session?.user.id
  useEffect(() => {
    if (!ready) return
    let alive = true
    if (!uid) {
      setProfile(null)
      setLoading(false)
      return
    }
    setLoading(true)
    supabase.from('dashboard_users').select('user_id,email,full_name,role').eq('user_id', uid).maybeSingle()
      .then(({ data }) => {
        if (!alive) return
        setProfile((data as DashboardUser) ?? null)
        setLoading(false)
      })
    return () => { alive = false }
  }, [uid, ready])

  const value: AuthState = {
    loading,
    session,
    profile,
    async signIn(email, password) {
      const { error } = await supabase.auth.signInWithPassword({ email, password })
      if (!error) return null
      if (error.message.toLowerCase().includes('invalid login')) return 'Email hoặc mật khẩu không đúng.'
      return error.message
    },
    async signOut() {
      await supabase.auth.signOut()
    },
  }
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useAuth(): AuthState {
  const v = useContext(Ctx)
  if (!v) throw new Error('useAuth ngoai AuthProvider')
  return v
}

export function displayName(p: DashboardUser | null, email?: string): string {
  return p?.full_name || (p?.email || email || '').split('@')[0] || 'Người dùng'
}

export function initials(name: string): string {
  const parts = name.trim().split(/\s+/)
  return ((parts[0]?.[0] || '') + (parts.length > 1 ? parts[parts.length - 1][0] : parts[0]?.[1] || '')).toUpperCase()
}
