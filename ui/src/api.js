// Thin client for the read-only ToolScope API. Every number shown in the UI comes from these responses.
async function request(path, options) {
  const res = await fetch(`/api${path}`, options)
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.detail || `${res.status} ${res.statusText}`)
  }
  return res.json()
}

const qs = (params) => new URLSearchParams(params).toString()

export const api = {
  metrics: () => request('/metrics'),
  agentRuns: () => request('/agent_runs'),
  taskSuccess: (variant = 'status') => request(`/task_success?${qs({ variant })}`),
  judge: (variant = 'status') => request(`/judge?${qs({ variant })}`),
  failures: (variant = 'status', diagnosis = 'v1') => request(`/failures?${qs({ variant, diagnosis })}`),
  cases: (agent, strategy, label, { offset = 0, limit = 20, variant = 'status', diagnosis = 'v1', reportId } = {}) =>
    request(`/cases?${qs({ agent, strategy, label, offset, limit, variant, diagnosis, paginated: true,
      ...(reportId ? { report_id: reportId } : {}) })}`),
  trajectory: (agent, strategy, taskId) => request(`/trajectory?${qs({ agent, strategy, task_id: taskId })}`),
  query: (taskId) => request(`/queries/${encodeURIComponent(taskId)}`),
  tools: (limit = 200) => request(`/tools?${qs({ limit })}`),
  tool: (toolId) => request(`/tools/${toolId}`),
  whatif: (toolId, description, taskIds) =>
    request('/whatif', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tool_id: toolId, description, task_ids: taskIds }),
    }),
}
