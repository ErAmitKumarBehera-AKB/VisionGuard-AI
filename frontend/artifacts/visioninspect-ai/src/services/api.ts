import type { DashboardStats, Feedback, Inspection, Model, InspectionStatus, Notification, Role, SystemStatus, User } from '@/data/types';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '/api/v1';
const cachedInspections: Inspection[] = [];
const cachedFeedback: Feedback[] = [];

const normalizeUser = (value: any): User => ({ id: value.id ?? value._id ?? '', name: value.name ?? value.full_name ?? '', email: value.email ?? '', role: 'USER', organization: value.organization ?? '', jobTitle: value.jobTitle ?? value.job_title ?? '', employeeId: value.employeeId ?? value.employee_id ?? '', plantName: value.plantName ?? value.plant_name ?? '', status: value.is_active === false ? 'SUSPENDED' : 'ACTIVE', createdAt: value.created_at ?? new Date().toISOString(), lastLogin: value.last_login ?? new Date().toISOString() });

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
  register(payload: { email: string; password: string; fullName: string; employeeId: string }) {
    return request<any>('/auth/register', { method: 'POST', body: JSON.stringify({ email: payload.email, password: payload.password, full_name: payload.fullName, employee_id: payload.employeeId }) });
  },
  login(email: string, password: string) {
    return request<any>('/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }).then(value => { localStorage.setItem('vi-token', value.access_token); return normalizeUser(value.user); });
  },
  current() {
    const token = localStorage.getItem('vi-token'); return token ? request<any>('/auth/me').then(normalizeUser) : Promise.resolve(null);
  },
  changePassword(currentPassword: string, newPassword: string) { return request<any>('/auth/me/password', { method: 'PUT', body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }) }); },
  updateProfile(payload: Partial<Pick<User, 'name' | 'organization' | 'jobTitle' | 'employeeId' | 'plantName'>>) {
    return request<any>('/auth/me', { method: 'PUT', body: JSON.stringify({ full_name: payload.name, organization: payload.organization, job_title: payload.jobTitle, employee_id: payload.employeeId, plant_name: payload.plantName }) }).then(normalizeUser);
  },
  logout() {
    localStorage.removeItem('vi-token'); return Promise.resolve();
  },
};

const normalizeInspection = (raw: any): Inspection => ({ id: raw.inspection_id ?? raw.id ?? '', timestamp: raw.timestamp ?? '', imageUrl: raw.image_url ?? `${API_BASE}/inspection-evidence/${raw.inspection_id ?? raw.id}`, product: raw.product_category ?? raw.product ?? 'Unclassified', prediction: raw.predicted_label ?? raw.prediction ?? 'UNKNOWN', confidence: Number(raw.confidence ?? 0), modelVersion: raw.model_version ?? 'unknown', inferenceTimeMs: Number(raw.inference_latency_ms ?? raw.latency_ms ?? 0), deviceId: raw.device_id ?? raw.machine_id ?? 'unknown', status: (raw.status === 'PENDING_REVIEW' || raw.status === 'FLAGGED' || raw.review_required) ? 'REVIEW' : (raw.final_label ?? raw.predicted_label ?? raw.prediction) === 'DEFECT' ? 'DEFECTIVE' : 'PASS', reviewed: Boolean(raw.review_completed), reviewRequired: Boolean(raw.review_required), finalLabel: raw.final_label ?? null, operatorFlagged: Boolean(raw.operator_flagged), remark: raw.operator_remark ?? null, registered: raw.registered !== false });

export const apiService = {
  async listInspections(role: Role = 'ADMINISTRATOR') {
    const raw = await request<any[]>('/admin/inspections');
    return raw.map(normalizeInspection);
  },
  async getInspection(id: string, role: Role = 'ADMINISTRATOR') { return normalizeInspection(await request<any>(`${'/admin/inspections'}/${id}`)); },
  async dashboard(role: Role = 'ADMINISTRATOR'): Promise<DashboardStats> {
    return request<DashboardStats>('/admin/dashboard');
  },
  async listModels(): Promise<Model[]> { return request<any[]>('/models').then(items => items.map(item => ({ version: item.model_name ? `${item.model_name} · ${item.model_version ?? 'unversioned'}` : (item.version ?? item.model_version ?? 'unknown'), architecture: item.architecture, status: item.status ?? 'UNKNOWN', datasetVersion: item.dataset_version ?? '', accuracy: Number(item.accuracy ?? 0), precision: Number(item.precision ?? 0), recall: Number(item.recall ?? 0), f1Score: Number(item.f1_score ?? item.f1Score ?? 0), latency: Number(item.latency ?? item.latency_ms ?? 0), createdAt: item.created_at ?? '', deploymentStatus: item.deployment_status ?? item.status ?? '' }))); },
  async systemStatus(): Promise<any> { return request<any>('/system/status'); },
  async pendingReviews(role: Role = 'ADMINISTRATOR'): Promise<any[]> { return request<any[]>('/feedback/pending').then(items => items.map(item => ({ ...item, image_url: item.image_url ?? `${API_BASE}/inspection-evidence/${item.inspection_id}` }))); },
  async machine(): Promise<any> { return request<any>('/admin/machines'); },
  async changePassword(currentPassword: string, newPassword: string) { return request<any>('/auth/me/password', { method: 'PUT', body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }) }); },
  async listMachines(): Promise<any[]> { return request<any[]>('/admin/machines'); },
  async listAuditLogs(): Promise<any[]> { return request<any[]>('/admin/audit-logs'); },
  async runInspection(file: File, productCategory = 'generic_part') {
    const body = new FormData();
    body.append('image', file, file.name);
    body.append('product_category', productCategory);
    const raw = await request<any>('/admin/inspect', { method: 'POST', headers: {}, body });
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
      registered: raw.registered !== false,
    };
    cachedInspections.unshift(result);
    return result;
  },
  async flagInspection(id: string) {
    const raw = await request<any>(`/admin/inspections/${id}/flag`, { method: 'POST' });
    const cached = cachedInspections.find(item => item.id === id);
    if (cached) Object.assign(cached, { status: 'REVIEW', reviewRequired: true, operatorFlagged: true, finalLabel: null });
    return raw;
  },
  async submitReview(id: string, humanLabel: 'OK' | 'DEFECT', remark: string) {
    const raw = await request<any>(`/admin/inspections/${id}/review`, { method: 'POST', body: JSON.stringify({ human_label: humanLabel, remark }) });
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
