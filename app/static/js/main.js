/**
 * MPLADS AI - Monitoring System Main JavaScript Utilities
 */

document.addEventListener("DOMContentLoaded", function () {
  // 1. Sidebar Toggle for Mobile
  const sidebarToggle = document.getElementById("sidebarToggle");
  const sidebar = document.getElementById("sidebar");

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener("click", function () {
      sidebar.classList.toggle("d-none");
    });
  }

  // 2. Auto-dismiss alerts after 5 seconds
  const autoAlerts = document.querySelectorAll(".alert-dismissible");
  autoAlerts.forEach(function (alert) {
    setTimeout(function () {
      try {
        const bsAlert = new bootstrap.Alert(alert);
        bsAlert.close();
      } catch (e) {}
    }, 5000);
  });

  // 3. Global Search Input Handler
  const globalSearch = document.getElementById("globalSearchInput");
  if (globalSearch) {
    globalSearch.addEventListener("keypress", function (e) {
      if (e.key === "Enter") {
        const query = globalSearch.value.trim();
        if (query.length > 0) {
          window.location.href = `/risk?q=${encodeURIComponent(query)}`;
        }
      }
    });
  }
});

/**
 * Updates alert status via AJAX
 */
function updateAlertStatus(workId, newStatus, selectElement) {
  const originalStatus = selectElement.getAttribute("data-original-status") || "Open";
  selectElement.disabled = true;

  fetch("/api/alerts/update", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      work_id: workId,
      status: newStatus,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      selectElement.disabled = false;
      if (data.success) {
        selectElement.setAttribute("data-original-status", newStatus);
        // Highlight updated row briefly
        const row = selectElement.closest("tr");
        if (row) {
          row.classList.add("table-success");
          setTimeout(() => row.classList.remove("table-success"), 1500);
        }
      } else {
        alert("Failed to update status: " + (data.message || "Unknown error"));
        selectElement.value = originalStatus;
      }
    })
    .catch((err) => {
      selectElement.disabled = false;
      alert("Error updating status: " + err);
      selectElement.value = originalStatus;
    });
}
