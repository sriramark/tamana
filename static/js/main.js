document.addEventListener("DOMContentLoaded", () => {
  const sidebar = document.getElementById("sidebar");
  const mobileToggle = document.getElementById("mobileToggle");
  const overlay = document.getElementById("sidebarOverlay");

  if (mobileToggle && sidebar) {
    mobileToggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
      if (overlay) overlay.classList.toggle("active");
    });
  }

  if (overlay) {
    overlay.addEventListener("click", () => {
      if (sidebar) sidebar.classList.remove("open");
      overlay.classList.remove("active");
    });
  }

  document.querySelectorAll("[data-download-img]").forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-download-img");
      const img = document.getElementById(targetId);
      if (!img) return;
      const a = document.createElement("a");
      a.href = img.src;
      a.download = (targetId || "chart") + ".png";
      a.click();
    });
  });

  document.querySelectorAll("form[data-loading]").forEach(form => {
    form.addEventListener("submit", () => {
      const btn = form.querySelector("button[type=submit]");
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing…`;
      }
    });
  });

  const arimaForm = document.getElementById("arimaOrderForm");
  if (arimaForm) {
    arimaForm.addEventListener("submit", e => {
      const p = parseInt(document.getElementById("p")?.value || 0);
      const d = parseInt(document.getElementById("d")?.value || 0);
      const q = parseInt(document.getElementById("q")?.value || 0);
      if (p < 0 || d < 0 || q < 0 || p > 5 || d > 2 || q > 5) {
        e.preventDefault();
        showToast("ARIMA order values: p,q in [0,5], d in [0,2]", "warning");
      }
    });
  }

  document.querySelectorAll(".flash-auto-dismiss").forEach(el => {
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transform = "translateY(-8px)";
      el.style.transition = "all 0.4s ease";
      setTimeout(() => el.remove(), 400);
    }, 4500);
  });

  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    const colors = { info: "#3b82f6", warning: "#f97316", success: "#10b981", danger: "#ef4444" };
    toast.style.cssText = `
      position:fixed; bottom:24px; right:24px; z-index:9999;
      background:#ffffff; border:1px solid ${colors[type] || colors.info};
      color:#14171c; padding:12px 20px; border-radius:8px;
      font-size:0.85rem; font-family:Inter,sans-serif;
      box-shadow:0 4px 16px rgba(20,23,28,0.14);
      max-width:320px;
    `;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = "0"; setTimeout(() => toast.remove(), 300); }, 3500);
  }
});
