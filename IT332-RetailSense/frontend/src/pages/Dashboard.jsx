"use client"

import { useState, useEffect } from "react"
import { Link } from "react-router-dom"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from "recharts"
import { Video, Map, Users, Clock, ArrowUp, ArrowDown } from "lucide-react"
import { heatmapService } from "../services/api"
import toast from "react-hot-toast"
import "../styles/dashboard.css"
import LoadingOverlay from "../components/LoadingOverlay"

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalVisitors: 0,
    averageDwellTime: 0,
    processedVideos: 0,
    generatedHeatmaps: 0,
    dwellTimeTrendChange: null,
  })
  const [dwellTimeTrend, setDwellTimeTrend] = useState([])
  const [visitorCount, setVisitorCount] = useState([])
  const [hourlyTrend, setHourlyTrend] = useState([])
  const [trendValues, setTrendValues] = useState({
    totalVisitors: 0,
    processedVideos: 0,
    generatedHeatmaps: 0,
  })
  const [isLoading, setIsLoading] = useState(true)
  const [exportLoading, setExportLoading] = useState(false)
  const [exportMenuOpen, setExportMenuOpen] = useState(false)
  const [visitorPeriod, setVisitorPeriod] = useState("monthly")
  const [hourlyPeriod, setHourlyPeriod] = useState("today")
  const [visitorCountFilter, setVisitorCountFilter] = useState("monthly")
  const [dwellTimeTrendFilter, setDwellTimeTrendFilter] = useState("daily")

  useEffect(() => {
    const fetchDashboardAnalytics = async () => {
      setIsLoading(true)
      try {
        const analytics = await heatmapService.getDashboardAnalytics()
        setStats({
          totalVisitors: analytics.total_visitors,
          averageDwellTime: analytics.average_dwell_time || 0,
          processedVideos: analytics.processed_videos || 0,
          generatedHeatmaps: analytics.generated_heatmaps || 0,
          dwellTimeTrendChange: analytics.dwell_time_trend_change ?? null,
        })
        let trend = []
        if (dwellTimeTrendFilter === "daily") trend = analytics.dwell_time_trend_daily || []
        else if (dwellTimeTrendFilter === "weekly") trend = analytics.dwell_time_trend_weekly || []
        else if (dwellTimeTrendFilter === "monthly") trend = analytics.dwell_time_trend_monthly || []
        setDwellTimeTrend(trend)
        let vCount = []
        if (analytics.visitor_count) {
          if (typeof analytics.visitor_count === "object") {
            vCount = analytics.visitor_count[visitorCountFilter] || []
          } else if (Array.isArray(analytics.visitor_count)) {
            vCount = analytics.visitor_count
          }
        }
        setVisitorCount(vCount)
        let hTrend = []
        if (analytics.hourly_trend) {
          if (Array.isArray(analytics.hourly_trend)) {
            hTrend = analytics.hourly_trend
          } else {
            hTrend = analytics.hourly_trend[hourlyPeriod] || []
          }
        }
        setHourlyTrend(hTrend)
      } catch (error) {
        console.error("Error fetching dashboard data:", error)
        toast.error("Failed to load dashboard data")
      } finally {
        setIsLoading(false)
      }
    }
    fetchDashboardAnalytics()
  }, [visitorPeriod, hourlyPeriod, visitorCountFilter, dwellTimeTrendFilter])

  const handleExportData = async (format) => {
    setExportLoading(true)
    setExportMenuOpen(false)
    try {
      const token = localStorage.getItem("access_token")
      if (!token) {
        toast.error("You must be logged in to export data.")
        setExportLoading(false)
        return
      }
      const response = await fetch(`/api/dashboard_export?format=${format}`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (response.status === 401) {
        toast.error("Session expired. Please log in again.")
        setExportLoading(false)
        return
      }
      if (!response.ok) throw new Error("Export failed")
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = format === "csv" ? "dashboard_export.csv" : "dashboard_export.pdf"
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      toast.error("Export failed")
    } finally {
      setExportLoading(false)
    }
  }

  return (
    <div>
      <LoadingOverlay
        show={isLoading || exportLoading}
        progressPercent={null}
        label={exportLoading ? "Exporting..." : "Loading dashboard..."}
      />
      <div className="dashboard-container">
        <div className="dashboard-header">
          <h1 className="dashboard-title">Overview</h1>
          <div className="dashboard-actions" style={{ position: "relative" }}>
            <button className="export-btn-modern" onClick={() => setExportMenuOpen((v) => !v)} >
              <svg
                style={{ marginRight: 8, verticalAlign: "middle" }}
                width="18"
                height="18"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="7 10 12 15 17 10" />
                <line x1="12" y1="15" x2="12" y2="3" />
              </svg>
              Export Data
            </button>
            {exportMenuOpen && (
              <div className="export-menu">
                <button className="export-menu-item" onClick={() => handleExportData("pdf")} style={{ color: '#000' }}>
                  Export as PDF
                </button>
                <button className="export-menu-item" onClick={() => handleExportData("csv")} style={{ color: '#000' }}>
                  Export as CSV
                </button>
              </div>
            )}
          </div>
        </div>

        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-content">
              <div className="stat-info">
                <p className="stat-value" style={{ color: '#000' }}>{isLoading ? "..." : stats.totalVisitors}</p>
                <p className="stat-label" style={{ color: '#000' }}>Total Visitors</p>
                <div className={`stat-trend ${trendValues.totalVisitors >= 0 ? "trend-up" : "trend-down"}`}>
                  {trendValues.totalVisitors >= 0 ? (
                    <ArrowUp className="trend-icon" />
                  ) : (
                    <ArrowDown className="trend-icon" />
                  )}
                  <span className="trend-value">{Math.abs(trendValues.totalVisitors).toFixed(1)}%</span>
                  <span className="trend-period">vs last week</span>
                </div>
              </div>
              <div className="stat-icon-container users-icon">
                <Users className="stat-icon" />
              </div>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-content">
              <div className="stat-info">
                <p className="stat-value" style={{ color: '#000' }}>{isLoading ? "..." : `${stats.averageDwellTime?.toFixed(1) || 0} min`}</p>
                <p className="stat-label" style={{ color: '#000' }}>Avg. Dwell Time</p>
                <div
                  className={`stat-trend ${
                    stats.dwellTimeTrendChange === null
                      ? ""
                      : stats.dwellTimeTrendChange >= 0
                        ? "trend-up"
                        : "trend-down"
                  }`}
                >
                  {stats.dwellTimeTrendChange === null ? null : stats.dwellTimeTrendChange >= 0 ? (
                    <ArrowUp className="trend-icon" />
                  ) : (
                    <ArrowDown className="trend-icon" />
                  )}
                  <span className="trend-value">
                    {stats.dwellTimeTrendChange === null || stats.dwellTimeTrendChange === undefined
                      ? "N/A"
                      : Math.abs(stats.dwellTimeTrendChange).toFixed(1) + "%"}
                  </span>
                  <span className="trend-period">vs last week</span>
                </div>
              </div>
              <div className="stat-icon-container clock-icon">
                <Clock className="stat-icon" />
              </div>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-content">
              <div className="stat-info">
                <p className="stat-value" style={{ color: '#000' }}>{isLoading ? "..." : stats.processedVideos}</p>
                <p className="stat-label" style={{ color: '#000' }}>Processed Videos</p>
                <div className={`stat-trend ${trendValues.processedVideos >= 0 ? "trend-up" : "trend-down"}`}>
                  {trendValues.processedVideos >= 0 ? (
                    <ArrowUp className="trend-icon" />
                  ) : (
                    <ArrowDown className="trend-icon" />
                  )}
                  <span className="trend-value">{Math.abs(trendValues.processedVideos).toFixed(1)}%</span>
                  <span className="trend-period">vs last month</span>
                </div>
              </div>
              <div className="stat-icon-container video-icon">
                <Video className="stat-icon" />
              </div>
            </div>
          </div>

          <div className="stat-card">
            <div className="stat-content">
              <div className="stat-info">
                <p className="stat-value" style={{ color: '#000' }}>{isLoading ? "..." : stats.generatedHeatmaps}</p>
                <p className="stat-label" style={{ color: '#000' }}>Generated Heatmaps</p>
                <div className={`stat-trend ${trendValues.generatedHeatmaps >= 0 ? "trend-up" : "trend-down"}`}>
                  {trendValues.generatedHeatmaps >= 0 ? (
                    <ArrowUp className="trend-icon" />
                  ) : (
                    <ArrowDown className="trend-icon" />
                  )}
                  <span className="trend-value">{Math.abs(trendValues.generatedHeatmaps).toFixed(1)}%</span>
                  <span className="trend-period">vs last month</span>
                </div>
              </div>
              <div className="stat-icon-container map-icon">
                <Map className="stat-icon" />
              </div>
            </div>
          </div>
        </div>

        <div className="dashboard-grid">
          <div>
            <div className="chart-card">
              <div className="chart-header">
                <h2 className="chart-title" style={{ color: '#000' }}>Average Dwell Time Trend</h2>
                <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                  <select value={dwellTimeTrendFilter} onChange={(e) => setDwellTimeTrendFilter(e.target.value)} style={{ color: '#000' }}>
                    <option value="daily">Daily</option>
                    <option value="weekly">Weekly</option>
                    <option value="monthly">Monthly</option>
                  </select>
                  <div className="chart-legend">
                    <div className="legend-item">
                      <span className="legend-color" style={{ backgroundColor: "#f97316" }}></span>
                      <span className="legend-label">Avg. Dwell Time</span>
                    </div>
                  </div>
                </div>
              </div>

              {isLoading ? (
                <div className="loading-indicator">
                  <div className="spinner"></div>
                </div>
              ) : (
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={dwellTimeTrend}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                      <XAxis
                        dataKey={
                          dwellTimeTrendFilter === "daily"
                            ? "date"
                            : dwellTimeTrendFilter === "weekly"
                              ? "week"
                              : "month"
                        }
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: "#6b7280", fontSize: 12 }}
                        label={{
                          value: dwellTimeTrendFilter.charAt(0).toUpperCase() + dwellTimeTrendFilter.slice(1),
                          position: "insideBottom",
                          offset: -5,
                          fill: "#6b7280",
                        }}
                      />
                      <YAxis
                        domain={[0, 60]}
                        tickFormatter={(v) => `${v} min`}
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: "#6b7280", fontSize: 12 }}
                      />
                      <Tooltip
                        formatter={(value) => [`${parseFloat(value).toFixed(2)} min`, "Avg. Dwell Time"]}
                      />

                      <Line
                        type="monotone"
                        dataKey="dwellTime"
                        stroke="#f97316"
                        strokeWidth={3}
                        dot={{ r: 6, strokeWidth: 2 }}
                        activeDot={{ r: 8, stroke: "#f97316", strokeWidth: 2 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            <div className="chart-card">
              <div className="chart-header">
                <h2 className="chart-title" style={{ color: '#000' }}>Visitor Count</h2>
                <select value={visitorCountFilter} onChange={(e) => setVisitorCountFilter(e.target.value)} style={{ color: '#000' }}>
                  <option value="daily">Daily</option>
                  <option value="weekly">Weekly</option>
                  <option value="monthly">Monthly</option>
                </select>
              </div>
              {isLoading ? (
                <div className="loading-indicator">
                  <div className="spinner"></div>
                </div>
              ) : (
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={visitorCount}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
                      <XAxis
                        dataKey="period"
                        axisLine={false}
                        tickLine={false}
                        tick={{ fill: "#6b7280", fontSize: 12 }}
                      />
                      <YAxis axisLine={false} tickLine={false} tick={{ fill: "#6b7280", fontSize: 12 }} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "white",
                          border: "none",
                          borderRadius: "8px",
                          boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.1)",
                          color: "#000",
                        }}
                        formatter={(value) => [`${value}`, "Visitors"]}
                      />
                      <Bar dataKey="count" fill="#4f46e5" radius={[4, 4, 0, 0]} barSize={30}  />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </div>

          <div>
            <div className="actions-card">
              <h2 className="actions-title" style={{ color: '#000' }}>Quick Actions</h2>
              <div className="actions-buttons">
                <Link to="/video-processing" className="action-btn video-btn">
                  <div className="action-icon-container video-icon-container">
                    <Video className="action-icon" />
                  </div>
                  <div className="action-text">
                    <p className="action-title">Process New Video</p>
                    <p className="action-description">Upload and analyze footage</p>
                  </div>
                </Link>

                <Link to="/heatmap-generation" className="action-btn heatmap-btn">
                  <div className="action-icon-container heatmap-icon-container">
                    <Map className="action-icon" />
                  </div>
                  <div className="action-text">
                    <p className="action-title">Generate Heatmap</p>
                    <p className="action-description">Create traffic visualization</p>
                  </div>
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
