import type { Feedback, Inspection, Notification, Role, User } from '@/data/types';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '/api/v1';
const cachedInspections: Inspection[] = [];
const cachedFeedback: Feedback[] = [];

const normalizeUser = (value: any): User => ({ id: value.id ?? value._id ?? '', name: value.name ?? value.full_name ?? '', email: value.email ?? '', role: value.role === 'ADMIN' ? 'ADMINISTRATOR' : value.role === 'SUPERVISOR' ? 'QUALITY_CONTROL_OPERATOR' : 'PUBLIC_USER', organization: value.organization ?? 'VisionInspect', jobTitle: value.jobTitle ?? value.role ?? '', status: value.is_active === false ? 'SUSPENDED' : 'ACTIVE', createdAt: value.created_at ?? new Date().toISOString(), lastLogin: value.last_login ?? new Date().toISOString() });

const request = async <T>(path: string, options?: RequestInit): Promise<T> => {
  const headers = new Headers(options?.headers);
  if (!(options?.body instanceof FormData)) headers.set('Content-Type', 'application/json');
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: (() => { const token = localStorage.getItem('vi-token'); if (token) headers.set('Authorization', `Bearer ${token}`); return headers; })(),
    credentials: 'include',
  });
  if (!response.ok) {
    let detail = '';
    try { detail = (await response.json()).detail ?? ''; } catch { /* non-JSON error */ }
    const messages: Record<number, string> = { 401: 'Your session has expired. Please login again.', 403: 'You do not have permission to access this resource.', 422: 'Please check the submitted information.', 503: 'AI inference service is currently unavailable.' };
    throw new Error(detail || messages[response.status] || 'Something went wrong. Please try again.');
  }
  return response.json() as Promise<T>;
};

export const authService = {
  login(email: string, password: string, role: Role) {
    return request<any>('/auth/login', { method: 'POST', body: JSON.stringify({ email, password, role }) }).then(value => { localStorage.setItem('vi-token', value.access_token); return normalizeUser(value.user); });
  },
  current() {
    const token = localStorage.getItem('vi-token'); return token ? request<any>('/auth/me').then(normalizeUser) : Promise.resolve(null);
  },
  logout() {
    localStorage.removeItem('vi-token'); return Promise.resolve();
  },
};

export const apiService = {
  listInspections() { return cachedInspections; },
  getInspection(id: string) { return cachedInspections.find(item => item.id === id); },
  async runInspection(file: File, productCategory = 'generic_part') {
    const body = new FormData();
    body.append('image', file, file.name);
    body.append('product_category', productCategory);
    const raw = await request<any>('/supervisor/inspect', { method: 'POST', headers: {}, body });
    const result: Inspection = {
      id: raw.inspection_id,
      timestamp: raw.timestamp,
      imageUrl: raw.image_url ?? '',
      product: raw.product_category ?? productCategory,
      prediction: raw.predicted_label ?? raw.prediction,
      confidence: raw.confidence,
      modelVersion: raw.model_version,
      inferenceTimeMs: raw.inference_latency_ms ?? raw.latency_ms ?? 0,
      deviceId: 'BROWSER / INSTA360 LINK 2',
      status: raw.status === 'PENDING_REVIEW' || raw.status === 'FLAGGED' ? 'REVIEW' : raw.final_label === 'DEFECT' ? 'DEFECTIVE' : 'PASS',
      reviewed: Boolean(raw.review_completed),
      reviewRequired: Boolean(raw.review_required),
      finalLabel: raw.final_label ?? null,
      operatorFlagged: Boolean(raw.operator_flagged),
      remark: raw.operator_remark ?? null,
    };
    cachedInspections.unshift(result);
    return result;
  },
  async flagInspection(id: string) {
    const raw = await request<any>(`/supervisor/inspections/${id}/flag`, { method: 'POST' });
    const cached = cachedInspections.find(item => item.id === id);
    if (cached) Object.assign(cached, { status: 'REVIEW', reviewRequired: true, operatorFlagged: true, finalLabel: null });
    return raw;
  },
  async submitReview(id: string, humanLabel: 'OK' | 'DEFECT', remark: string) {
    const raw = await request<any>(`/supervisor/inspections/${id}/review`, { method: 'POST', body: JSON.stringify({ human_label: humanLabel, remark }) });
    const cached = cachedInspections.find(item => item.id === id);
    if (cached) Object.assign(cached, { status: humanLabel === 'DEFECT' ? 'DEFECTIVE' : 'PASS', reviewed: true, reviewRequired: true, finalLabel: humanLabel, remark });
    return raw;
  },
  addFeedback(item: Feedback) {
    cachedFeedback.unshift(item); return Promise.resolve(item);
  },
  listFeedback() {
    return cachedFeedback;
  },
  listNotifications() {
    return request<Notification[]>('/notifications');
  },
  markNotificationsRead() {
    return request<void>('/notifications/read', { method: 'POST' });
  },
};
