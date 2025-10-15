PII_INDICATOR_STYLE = """
<style>
.pii-indicators {
  display: grid;
  gap: 0.75rem;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  margin-bottom: 0.5rem;
}
.pii-indicator {
  background: var(--secondary-background-color, rgba(250, 250, 250, 0.85));
  border-radius: 0.75rem;
  border: 1px solid rgba(49, 51, 63, 0.15);
  box-shadow: 0 2px 6px rgba(15, 23, 42, 0.08);
  padding: 0.75rem 1rem;
  text-align: center;
}
.pii-indicator__label {
  color: rgba(49, 51, 63, 0.65);
  font-size: 0.75rem;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.pii-indicator__value {
  color: var(--text-color, #0d0d0d);
  font-size: 1.75rem;
  font-weight: 600;
  margin-top: 0.35rem;
}
</style>
"""

GUARDRAIL_PANEL_STYLE = """
<style>
.guardrail-panel { display: grid; gap: 1.5rem; }
.guardrail-panel__chart {
  background: var(--secondary-background-color, rgba(248, 250, 252, 0.85));
  border-radius: 1rem;
  border: 1px solid rgba(148, 163, 184, 0.35);
  padding: 1rem;
}
.guardrail-card-list {
  max-height: 320px; overflow-y: auto; padding-right: 0.5rem;
  display: grid; gap: 0.75rem;
}
.guardrail-card {
  background: rgba(255, 255, 255, 0.82);
  border-radius: 0.85rem;
  border: 1px solid rgba(148, 163, 184, 0.4);
  box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
  padding: 0.85rem 1rem;
}
.guardrail-card__subject {
  font-weight: 600; color: var(--text-color, #111827);
  margin-bottom: 0.35rem; line-height: 1.3;
}
.guardrail-card__meta {
  display: flex; align-items: center; justify-content: space-between;
  font-size: 0.8rem; color: rgba(55, 65, 81, 0.85);
  margin-bottom: 0.5rem; gap: 0.75rem;
}
</style>
"""
