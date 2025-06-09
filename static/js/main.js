import { Chart } from "@/components/ui/chart"

// Global variables
let currentResults = []
let currentPage = 1
const itemsPerPage = 10
let elbowChart = null
let silhouetteChart = null
const clustersData = []
const currentMode = "explore"
let animationId = null
let canvas, ctx

// Interactive background variables
const particles = []
const mouse = { x: 0, y: 0 }

// Sample cluster data
const sampleClusters = [
  {
    id: 1,
    name: "Paysages",
    color: "rgba(34, 197, 94, 0.8)",
    size: 120,
    position: { x: 20, y: 30 },
    images: [
      "/placeholder.svg?height=60&width=60&text=Paysage1",
      "/placeholder.svg?height=60&width=60&text=Paysage2",
      "/placeholder.svg?height=60&width=60&text=Paysage3",
      "/placeholder.svg?height=60&width=60&text=Paysage4",
    ],
    count: 24,
  },
  {
    id: 2,
    name: "Portraits",
    color: "rgba(59, 130, 246, 0.8)",
    size: 100,
    position: { x: 70, y: 20 },
    images: [
      "/placeholder.svg?height=60&width=60&text=Portrait1",
      "/placeholder.svg?height=60&width=60&text=Portrait2",
      "/placeholder.svg?height=60&width=60&text=Portrait3",
      "/placeholder.svg?height=60&width=60&text=Portrait4",
    ],
    count: 18,
  },
  {
    id: 3,
    name: "Architecture",
    color: "rgba(168, 85, 247, 0.8)",
    size: 90,
    position: { x: 15, y: 70 },
    images: [
      "/placeholder.svg?height=60&width=60&text=Archi1",
      "/placeholder.svg?height=60&width=60&text=Archi2",
      "/placeholder.svg?height=60&width=60&text=Archi3",
      "/placeholder.svg?height=60&width=60&text=Archi4",
    ],
    count: 15,
  },
  {
    id: 4,
    name: "Animaux",
    color: "rgba(249, 115, 22, 0.8)",
    size: 110,
    position: { x: 80, y: 65 },
    images: [
      "/placeholder.svg?height=60&width=60&text=Animal1",
      "/placeholder.svg?height=60&width=60&text=Animal2",
      "/placeholder.svg?height=60&width=60&text=Animal3",
      "/placeholder.svg?height=60&width=60&text=Animal4",
    ],
    count: 21,
  },
  {
    id: 5,
    name: "Objets",
    color: "rgba(14, 165, 233, 0.8)",
    size: 80,
    position: { x: 50, y: 50 },
    images: [
      "/placeholder.svg?height=60&width=60&text=Objet1",
      "/placeholder.svg?height=60&width=60&text=Objet2",
      "/placeholder.svg?height=60&width=60&text=Objet3",
      "/placeholder.svg?height=60&width=60&text=Objet4",
    ],
    count: 12,
  },
]

// Enhanced Interactive background with AI/ML theme
class AIParticle {
  constructor() {
    this.x = Math.random() * window.innerWidth
    this.y = Math.random() * window.innerHeight
    this.vx = (Math.random() - 0.5) * 0.8
    this.vy = (Math.random() - 0.5) * 0.8
    this.radius = Math.random() * 3 + 1
    this.opacity = Math.random() * 0.6 + 0.2
    this.color = this.getRandomColor()
    this.cluster = Math.floor(Math.random() * 5) // 5 clusters
    this.targetX = this.x
    this.targetY = this.y
    this.pulsePhase = Math.random() * Math.PI * 2
  }

  getRandomColor() {
    const colors = [
      "rgba(34, 197, 94, ", // Green - Success/Data
      "rgba(59, 130, 246, ", // Blue - Primary/Neural
      "rgba(168, 85, 247, ", // Purple - AI/ML
      "rgba(249, 115, 22, ", // Orange - Processing
      "rgba(14, 165, 233, ", // Sky Blue - Analysis
    ]
    return colors[Math.floor(Math.random() * colors.length)]
  }

  update() {
    // Clustering behavior - particles move toward cluster centers
    const clusterCenters = [
      { x: window.innerWidth * 0.2, y: window.innerHeight * 0.3 },
      { x: window.innerWidth * 0.8, y: window.innerHeight * 0.2 },
      { x: window.innerWidth * 0.5, y: window.innerHeight * 0.5 },
      { x: window.innerWidth * 0.3, y: window.innerHeight * 0.7 },
      { x: window.innerWidth * 0.7, y: window.innerHeight * 0.8 },
    ]

    const center = clusterCenters[this.cluster]
    const dx = center.x - this.x
    const dy = center.y - this.y
    const distance = Math.sqrt(dx * dx + dy * dy)

    // Gentle attraction to cluster center
    if (distance > 50) {
      this.vx += (dx / distance) * 0.001
      this.vy += (dy / distance) * 0.001
    }

    // Mouse interaction - stronger effect
    const mouseDx = mouse.x - this.x
    const mouseDy = mouse.y - this.y
    const mouseDistance = Math.sqrt(mouseDx * mouseDx + mouseDy * mouseDy)

    if (mouseDistance < 150) {
      const force = (150 - mouseDistance) / 150
      this.vx += (mouseDx / mouseDistance) * force * 0.02
      this.vy += (mouseDy / mouseDistance) * force * 0.02
    }

    this.x += this.vx
    this.y += this.vy

    // Bounce off edges
    if (this.x < 0 || this.x > canvas.width) this.vx *= -0.8
    if (this.y < 0 || this.y > canvas.height) this.vy *= -0.8

    // Limit velocity
    this.vx = Math.max(-2, Math.min(2, this.vx))
    this.vy = Math.max(-2, Math.min(2, this.vy))

    // Apply friction
    this.vx *= 0.99
    this.vy *= 0.99

    // Pulsing effect
    this.pulsePhase += 0.05
  }

  draw() {
    const pulse = Math.sin(this.pulsePhase) * 0.3 + 0.7
    const currentRadius = this.radius * pulse
    const currentOpacity = this.opacity * pulse

    // Main particle
    ctx.beginPath()
    ctx.arc(this.x, this.y, currentRadius, 0, Math.PI * 2)
    ctx.fillStyle = this.color + currentOpacity + ")"
    ctx.fill()

    // Glow effect
    ctx.beginPath()
    ctx.arc(this.x, this.y, currentRadius * 2, 0, Math.PI * 2)
    ctx.fillStyle = this.color + currentOpacity * 0.2 + ")"
    ctx.fill()
  }
}

// Particle class
// class Particle {
//   constructor() {
//     this.x = Math.random() * window.innerWidth
//     this.y = Math.random() * window.innerHeight
//     this.vx = (Math.random() - 0.5) * 0.5
//     this.vy = (Math.random() - 0.5) * 0.5
//     this.radius = Math.random() * 2 + 1
//     this.opacity = Math.random() * 0.5 + 0.2
//   }

//   update() {
//     this.x += this.vx
//     this.y += this.vy

//     // Bounce off edges
//     if (this.x < 0 || this.x > canvas.width) this.vx *= -1
//     if (this.y < 0 || this.y > canvas.height) this.vy *= -1

//     // Mouse interaction
//     const dx = mouse.x - this.x
//     const dy = mouse.y - this.y
//     const distance = Math.sqrt(dx * dx + dy * dy)

//     if (distance < 100) {
//       const force = (100 - distance) / 100
//       this.vx += (dx / distance) * force * 0.01
//       this.vy += (dy / distance) * force * 0.01
//     }

//     // Limit velocity
//     this.vx = Math.max(-2, Math.min(2, this.vx))
//     this.vy = Math.max(-2, Math.min(2, this.vy))
//   }

//   draw() {
//     ctx.beginPath()
//     ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2)
//     ctx.fillStyle = `rgba(255, 255, 255, ${this.opacity})`
//     ctx.fill()
//   }
// }

// Neural network connections
function drawNeuralConnections() {
  for (let i = 0; i < particles.length; i++) {
    for (let j = i + 1; j < particles.length; j++) {
      const dx = particles[i].x - particles[j].x
      const dy = particles[i].y - particles[j].y
      const distance = Math.sqrt(dx * dx + dy * dy)

      // Connect particles in the same cluster
      if (particles[i].cluster === particles[j].cluster && distance < 100) {
        const opacity = ((100 - distance) / 100) * 0.4

        ctx.beginPath()
        ctx.moveTo(particles[i].x, particles[i].y)
        ctx.lineTo(particles[j].x, particles[j].y)

        // Gradient line based on cluster
        const gradient = ctx.createLinearGradient(particles[i].x, particles[i].y, particles[j].x, particles[j].y)
        gradient.addColorStop(0, particles[i].color + opacity + ")")
        gradient.addColorStop(1, particles[j].color + opacity + ")")

        ctx.strokeStyle = gradient
        ctx.lineWidth = 2
        ctx.stroke()
      }
    }
  }
}

// Data flow visualization
function drawDataFlow() {
  const time = Date.now() * 0.001

  // Horizontal data streams
  for (let i = 0; i < 3; i++) {
    const y = (window.innerHeight / 4) * (i + 1)
    const x = (Math.sin(time + i) * 0.5 + 0.5) * window.innerWidth

    ctx.beginPath()
    ctx.arc(x, y, 4, 0, Math.PI * 2)
    ctx.fillStyle = `rgba(34, 197, 94, ${Math.sin(time * 2 + i) * 0.3 + 0.7})`
    ctx.fill()

    // Trail effect
    for (let j = 1; j < 10; j++) {
      const trailX = x - j * 15
      const trailOpacity = ((10 - j) / 10) * 0.3

      ctx.beginPath()
      ctx.arc(trailX, y, 2, 0, Math.PI * 2)
      ctx.fillStyle = `rgba(59, 130, 246, ${trailOpacity})`
      ctx.fill()
    }
  }
}

// Initialize canvas for background effects
function initializeCanvas() {
  canvas = document.getElementById("clusters-canvas")
  ctx = canvas.getContext("2d")
  resizeCanvas()

  window.addEventListener("resize", resizeCanvas)
}

function resizeCanvas() {
  canvas.width = window.innerWidth
  canvas.height = window.innerHeight
}

// Create floating cluster bubbles
function createFloatingClusters() {
  const container = document.getElementById("floating-clusters")
  container.innerHTML = ""

  sampleClusters.forEach((cluster, index) => {
    const bubble = document.createElement("div")
    bubble.className = "cluster-bubble"
    bubble.style.width = cluster.size + "px"
    bubble.style.height = cluster.size + "px"
    bubble.style.background = cluster.color
    bubble.style.left = cluster.position.x + "%"
    bubble.style.top = cluster.position.y + "%"
    bubble.style.animationDelay = index * 0.5 + "s"

    // Add floating animation
    bubble.style.animation = `floatCluster 6s ease-in-out infinite ${index * 0.5}s`

    const content = document.createElement("div")
    content.className = "cluster-content"

    const imagesGrid = document.createElement("div")
    imagesGrid.className = "cluster-images"

    cluster.images.slice(0, 4).forEach((imageSrc) => {
      const img = document.createElement("img")
      img.src = imageSrc
      img.className = "cluster-image"
      img.alt = cluster.name
      imagesGrid.appendChild(img)
    })

    const label = document.createElement("div")
    label.className = "cluster-label"
    label.textContent = cluster.name

    const count = document.createElement("div")
    count.className = "cluster-count"
    count.textContent = `${cluster.count} images`

    content.appendChild(imagesGrid)
    content.appendChild(label)
    content.appendChild(count)
    bubble.appendChild(content)

    // Add click event
    bubble.addEventListener("click", () => openClusterModal(cluster))

    container.appendChild(bubble)
  })
}

// Add floating animation CSS
const style = document.createElement("style")
style.textContent = `
  @keyframes floatCluster {
    0%, 100% {
      transform: translateY(0px) rotate(0deg);
    }
    33% {
      transform: translateY(-20px) rotate(2deg);
    }
    66% {
      transform: translateY(10px) rotate(-1deg);
    }
  }
`
document.head.appendChild(style)

// Background animation
function startAnimation() {
  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Draw connecting lines between clusters
    drawClusterConnections()

    // Draw floating particles
    drawFloatingParticles()

    animationId = requestAnimationFrame(animate)
  }
  animate()
}

function drawClusterConnections() {
  const clusters = document.querySelectorAll(".cluster-bubble")

  clusters.forEach((cluster1, i) => {
    clusters.forEach((cluster2, j) => {
      if (i < j) {
        const rect1 = cluster1.getBoundingClientRect()
        const rect2 = cluster2.getBoundingClientRect()

        const x1 = rect1.left + rect1.width / 2
        const y1 = rect1.top + rect1.height / 2
        const x2 = rect2.left + rect2.width / 2
        const y2 = rect2.top + rect2.height / 2

        const distance = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        if (distance < 300) {
          const opacity = ((300 - distance) / 300) * 0.3

          ctx.beginPath()
          ctx.moveTo(x1, y1)
          ctx.lineTo(x2, y2)
          ctx.strokeStyle = `rgba(102, 126, 234, ${opacity})`
          ctx.lineWidth = 2
          ctx.stroke()
        }
      }
    })
  })
}

function drawFloatingParticles() {
  const time = Date.now() * 0.001

  for (let i = 0; i < 20; i++) {
    const x = (Math.sin(time + i) * 0.3 + 0.5) * canvas.width
    const y = (Math.cos(time * 0.7 + i) * 0.3 + 0.5) * canvas.height
    const size = Math.sin(time * 2 + i) * 2 + 3
    const opacity = Math.sin(time * 1.5 + i) * 0.3 + 0.4

    ctx.beginPath()
    ctx.arc(x, y, size, 0, Math.PI * 2)
    ctx.fillStyle = `rgba(102, 126, 234, ${opacity})`
    ctx.fill()
  }
}

// Event listeners setup
function initializeEventListeners() {
  // Parameter sliders
  document.getElementById("cluster-count").addEventListener("input", function () {
    this.nextElementSibling.textContent = this.value
  })

  document.getElementById("sensitivity").addEventListener("input", function () {
    this.nextElementSibling.textContent = this.value
  })

  // File upload
  document.getElementById("image-files").addEventListener("change", handleFileUpload)

  // Upload zone drag and drop
  const uploadZone = document.getElementById("upload-zone")
  uploadZone.addEventListener("dragover", handleDragOver)
  uploadZone.addEventListener("dragleave", handleDragLeave)
  uploadZone.addEventListener("drop", handleDrop)
  uploadZone.addEventListener("click", () => document.getElementById("image-files").click())

  // Weight sliders
  const sliders = document.querySelectorAll(".slider")
  sliders.forEach((slider) => {
    slider.addEventListener("input", updateWeightDisplay)
  })

  // Test image input
  document.getElementById("test-image").addEventListener("change", handleTestImageSelect)

  // Search and sort
  document.getElementById("search-filter").addEventListener("input", filterResults)
  document.getElementById("sort-by").addEventListener("change", sortResults)
}

// Navigation functions
function goBack() {
  showToast("Retour à la page précédente", "info")
}

function openInNewTab() {
  showToast("Ouverture dans un nouvel onglet", "info")
}

// Main interaction functions
function exploreMode() {
  const configPanel = document.getElementById("config-panel")
  configPanel.classList.add("open")
}

function closeConfig() {
  const configPanel = document.getElementById("config-panel")
  configPanel.classList.remove("open")
}

function selectImages() {
  document.getElementById("image-files").click()
}

function handleFileUpload(event) {
  const files = event.target.files
  if (files.length > 0) {
    showToast(`${files.length} image(s) sélectionnée(s)`, "success")
    // Process files here
  }
}

function handleDragOver(event) {
  event.preventDefault()
  event.currentTarget.classList.add("dragover")
}

function handleDragLeave(event) {
  event.preventDefault()
  event.currentTarget.classList.remove("dragover")
}

function handleDrop(event) {
  event.preventDefault()
  event.currentTarget.classList.remove("dragover")

  const files = event.dataTransfer.files
  if (files.length > 0) {
    showToast(`${files.length} image(s) ajoutée(s)`, "success")
    // Process files here
  }
}

function startClustering() {
  closeConfig()
  showLoading()

  // Simulate clustering process
  setTimeout(() => {
    hideLoading()
    showToast("Clustering terminé avec succès!", "success")
    // Update clusters with new data
    updateClusters()
  }, 3000)
}

function showLoading() {
  document.getElementById("loading-overlay").style.display = "flex"

  // Animate progress bar
  const progressFill = document.getElementById("progress-fill")
  let progress = 0
  const interval = setInterval(() => {
    progress += Math.random() * 15
    if (progress >= 100) {
      progress = 100
      clearInterval(interval)
    }
    progressFill.style.width = progress + "%"
  }, 150)
}

function hideLoading() {
  document.getElementById("loading-overlay").style.display = "none"
}

function updateClusters() {
  // Add some animation to existing clusters
  const clusters = document.querySelectorAll(".cluster-bubble")
  clusters.forEach((cluster, index) => {
    setTimeout(() => {
      cluster.style.transform = "scale(1.2)"
      setTimeout(() => {
        cluster.style.transform = "scale(1)"
      }, 300)
    }, index * 100)
  })
}

// Cluster modal functions
function openClusterModal(cluster) {
  const modal = document.getElementById("cluster-modal")
  const title = document.getElementById("modal-title")
  const body = document.getElementById("modal-body")

  title.textContent = `Cluster: ${cluster.name}`

  body.innerHTML = `
    <div style="margin-bottom: 2rem;">
      <h4 style="margin-bottom: 1rem; color: #374151;">Informations du cluster</h4>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem;">
        <div style="background: #f9fafb; padding: 1rem; border-radius: 8px;">
          <div style="font-weight: 600; color: #6b7280; font-size: 0.9rem;">Nombre d'images</div>
          <div style="font-size: 1.5rem; font-weight: 700; color: #1f2937;">${cluster.count}</div>
        </div>
        <div style="background: #f9fafb; padding: 1rem; border-radius: 8px;">
          <div style="font-weight: 600; color: #6b7280; font-size: 0.9rem;">Similarité moyenne</div>
          <div style="font-size: 1.5rem; font-weight: 700; color: #1f2937;">87%</div>
        </div>
      </div>
    </div>
    
    <div>
      <h4 style="margin-bottom: 1rem; color: #374151;">Images du cluster</h4>
      <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 1rem;">
        ${cluster.images
          .map(
            (img) => `
          <div style="aspect-ratio: 1; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <img src="${img}" alt="${cluster.name}" style="width: 100%; height: 100%; object-fit: cover;">
          </div>
        `,
          )
          .join("")}
      </div>
    </div>
  `

  modal.style.display = "flex"
}

function closeModal() {
  document.getElementById("cluster-modal").style.display = "none"
}

// Handle test image selection
function selectTestImage() {
  document.getElementById("test-image").click()
}

function handleTestImageSelect(e) {
  const file = e.target.files[0]
  if (file) {
    handleTestImageFile(file)
  }
}

function handleTestImageFile(file) {
  if (!file.type.startsWith("image/")) {
    showToast("Veuillez sélectionner un fichier image valide.", "error")
    return
  }

  const reader = new FileReader()
  reader.onload = (e) => {
    const dropContent = document.querySelector(".drop-content")
    const dropPreview = document.getElementById("drop-preview")
    const previewImg = document.getElementById("preview-img")
    const previewName = document.getElementById("preview-name")

    previewImg.src = e.target.result
    previewName.textContent = file.name

    dropContent.style.display = "none"
    dropPreview.style.display = "flex"
  }
  reader.readAsDataURL(file)
}

function removeTestImage() {
  const dropContent = document.querySelector(".drop-content")
  const dropPreview = document.getElementById("drop-preview")
  const testImageInput = document.getElementById("test-image")

  dropContent.style.display = "flex"
  dropPreview.style.display = "none"
  testImageInput.value = ""
}

// Browse folder (simulation)
function browseFolder() {
  showToast("Fonctionnalité de navigation des dossiers simulée. Entrez le chemin manuellement.", "info")
}

// Start processing
function startProcessing(event) {
  // Validation
  const trainingPath = document.getElementById("training-path").value.trim()
  const testImage = document.getElementById("test-image").files[0]

  if (!trainingPath) {
    showToast("Veuillez spécifier le chemin du dossier training_set.", "error")
    return
  }

  if (!testImage) {
    showToast("Veuillez sélectionner une image de test.", "error")
    return
  }

  // Check if at least one descriptor is selected
  const descriptors = ["color", "shape", "texture", "vgg19"]
  const selectedDescriptors = descriptors.filter((desc) => document.getElementById(`desc-${desc}`).checked)

  if (selectedDescriptors.length === 0) {
    showToast("Veuillez sélectionner au moins un descripteur.", "error")
    return
  }

  // Add particle burst effect
  const button = event.target
  const rect = button.getBoundingClientRect()
  const x = rect.left + rect.width / 2
  const y = rect.top + rect.height / 2
  createParticleBurst(x, y)

  // Show loading
  showLoadingOverlay()

  // Simulate processing
  simulateProcessing()
}

// Show loading overlay
function showLoadingOverlay() {
  document.getElementById("loading-overlay").style.display = "flex"

  // Animate progress bar
  const progressFill = document.getElementById("progress-fill")
  let progress = 0
  const interval = setInterval(() => {
    progress += Math.random() * 15
    if (progress >= 100) {
      progress = 100
      clearInterval(interval)
    }
    progressFill.style.width = progress + "%"
  }, 200)
}

// Hide loading overlay
function hideLoadingOverlay() {
  document.getElementById("loading-overlay").style.display = "none"
}

// Simulate processing and show results
function simulateProcessing() {
  setTimeout(() => {
    hideLoadingOverlay()
    showResults()
    showToast("Analyse terminée avec succès !", "success")
  }, 3000)
}

// Show results section
function showResults() {
  const resultsPanel = document.getElementById("results-panel")
  resultsPanel.style.display = "block"
  resultsPanel.scrollIntoView({ behavior: "smooth" })

  // Generate charts
  generateElbowChart()
  generateSilhouetteChart()

  // Generate cluster gallery
  generateClusterGallery()

  // Generate KNN results
  generateKNNResults()
}

// Generate elbow chart
function generateElbowChart() {
  const ctx = document.getElementById("elbow-chart").getContext("2d")

  if (elbowChart) {
    elbowChart.destroy()
  }

  const k = Number.parseInt(document.getElementById("k-clusters").value)
  const kValues = Array.from({ length: k - 1 }, (_, i) => i + 2)
  const inertiaValues = kValues.map((k) => Math.random() * 1000 + 500 - k * 50)

  elbowChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: kValues,
      datasets: [
        {
          label: "Inertie",
          data: inertiaValues,
          borderColor: "#74b9ff",
          backgroundColor: "rgba(116, 185, 255, 0.1)",
          borderWidth: 3,
          pointBackgroundColor: "#74b9ff",
          pointBorderColor: "#ffffff",
          pointBorderWidth: 2,
          pointRadius: 6,
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: false,
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Nombre de clusters (k)",
          },
        },
        y: {
          title: {
            display: true,
            text: "Inertie",
          },
        },
      },
    },
  })
}

// Generate silhouette chart
function generateSilhouetteChart() {
  const ctx = document.getElementById("silhouette-chart").getContext("2d")

  if (silhouetteChart) {
    silhouetteChart.destroy()
  }

  const k = Number.parseInt(document.getElementById("k-clusters").value)
  const clusters = Array.from({ length: k }, (_, i) => i)
  const silhouetteValues = clusters.map(() => Math.random() * 0.8 + 0.2)

  // Update silhouette score display
  const avgScore = silhouetteValues.reduce((a, b) => a + b, 0) / silhouetteValues.length
  document.getElementById("silhouette-score").textContent = avgScore.toFixed(3)

  silhouetteChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: clusters.map((c) => `Cluster ${c}`),
      datasets: [
        {
          label: "Coefficient de silhouette",
          data: silhouetteValues,
          backgroundColor: clusters.map((_, i) => `hsl(${(i * 360) / k}, 70%, 60%)`),
          borderColor: clusters.map((_, i) => `hsl(${(i * 360) / k}, 70%, 50%)`),
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: {
          display: false,
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: "Clusters",
          },
        },
        y: {
          title: {
            display: true,
            text: "Coefficient de silhouette",
          },
          min: 0,
          max: 1,
        },
      },
    },
  })
}

// Generate cluster gallery
function generateClusterGallery() {
  const clustersGrid = document.getElementById("clusters-grid")
  const k = Number.parseInt(document.getElementById("k-clusters").value)

  clustersGrid.innerHTML = ""

  for (let i = 0; i < k; i++) {
    const clusterItem = document.createElement("div")
    clusterItem.className = "cluster-item"

    const clusterHeader = document.createElement("div")
    clusterHeader.className = "cluster-header"
    clusterHeader.textContent = `Cluster ${i} (${Math.floor(Math.random() * 20 + 5)} images)`

    const clusterImages = document.createElement("div")
    clusterImages.className = "cluster-images"

    // Generate 3 sample images per cluster
    for (let j = 0; j < 3; j++) {
      const img = document.createElement("img")
      img.className = "cluster-image"
      img.src = `/placeholder.svg?height=80&width=80&text=Img${i}-${j}`
      img.alt = `Cluster ${i} Image ${j}`
      clusterImages.appendChild(img)
    }

    clusterItem.appendChild(clusterHeader)
    clusterItem.appendChild(clusterImages)
    clustersGrid.appendChild(clusterItem)
  }
}

// Generate KNN results
function generateKNNResults() {
  const nNeighbors = Number.parseInt(document.getElementById("n-neighbors").value)
  const k = Number.parseInt(document.getElementById("k-clusters").value)

  currentResults = []

  for (let i = 0; i < nNeighbors; i++) {
    currentResults.push({
      rank: i + 1,
      filename: `image_${String(i + 1).padStart(3, "0")}.jpg`,
      cluster: Math.floor(Math.random() * k),
      distance: Math.random() * 2,
      imagePath: `/placeholder.svg?height=60&width=60&text=Img${i + 1}`,
    })
  }

  // Sort by distance initially
  currentResults.sort((a, b) => a.distance - b.distance)

  displayResults()
}

// Display results in table
function displayResults() {
  const tbody = document.getElementById("results-tbody")
  const startIndex = (currentPage - 1) * itemsPerPage
  const endIndex = startIndex + itemsPerPage
  const pageResults = currentResults.slice(startIndex, endIndex)

  tbody.innerHTML = ""

  pageResults.forEach((result) => {
    const row = document.createElement("tr")
    row.innerHTML = `
            <td>${result.rank}</td>
            <td><img src="${result.imagePath}" alt="${result.filename}" class="result-image"></td>
            <td>${result.filename}</td>
            <td><span class="cluster-badge">Cluster ${result.cluster}</span></td>
            <td>${result.distance.toFixed(3)}</td>
        `
    tbody.appendChild(row)
  })

  updatePagination()
}

// Update pagination
function updatePagination() {
  const pagination = document.getElementById("pagination")
  const totalPages = Math.ceil(currentResults.length / itemsPerPage)

  pagination.innerHTML = ""

  // Previous button
  const prevBtn = document.createElement("button")
  prevBtn.textContent = "‹ Précédent"
  prevBtn.disabled = currentPage === 1
  prevBtn.onclick = () => {
    if (currentPage > 1) {
      currentPage--
      displayResults()
    }
  }
  pagination.appendChild(prevBtn)

  // Page numbers
  for (let i = 1; i <= totalPages; i++) {
    const pageBtn = document.createElement("button")
    pageBtn.textContent = i
    pageBtn.className = i === currentPage ? "active" : ""
    pageBtn.onclick = () => {
      currentPage = i
      displayResults()
    }
    pagination.appendChild(pageBtn)
  }

  // Next button
  const nextBtn = document.createElement("button")
  nextBtn.textContent = "Suivant ›"
  nextBtn.disabled = currentPage === totalPages
  nextBtn.onclick = () => {
    if (currentPage < totalPages) {
      currentPage++
      displayResults()
    }
  }
  pagination.appendChild(nextBtn)
}

// Filter results
function filterResults() {
  const searchTerm = document.getElementById("search-filter").value.toLowerCase()
  const filteredResults = currentResults.filter((result) => result.filename.toLowerCase().includes(searchTerm))

  // Update display with filtered results
  const tbody = document.getElementById("results-tbody")
  tbody.innerHTML = ""

  filteredResults.forEach((result) => {
    const row = document.createElement("tr")
    row.innerHTML = `
            <td>${result.rank}</td>
            <td><img src="${result.imagePath}" alt="${result.filename}" class="result-image"></td>
            <td>${result.filename}</td>
            <td><span class="cluster-badge">Cluster ${result.cluster}</span></td>
            <td>${result.distance.toFixed(3)}</td>
        `
    tbody.appendChild(row)
  })
}

// Sort results
function sortResults() {
  const sortBy = document.getElementById("sort-by").value

  currentResults.sort((a, b) => {
    switch (sortBy) {
      case "distance":
        return a.distance - b.distance
      case "cluster":
        return a.cluster - b.cluster
      case "filename":
        return a.filename.localeCompare(b.filename)
      default:
        return 0
    }
  })

  currentPage = 1
  displayResults()
}

// Reset form
function resetForm() {
  // Reset inputs
  document.getElementById("training-path").value = ""
  document.getElementById("k-clusters").value = "5"
  document.getElementById("n-neighbors").value = "10"

  // Reset descriptors
  const descriptors = ["color", "shape", "texture", "vgg19"]
  descriptors.forEach((desc) => {
    document.getElementById(`desc-${desc}`).checked = true
    document.getElementById(`weight-${desc}`).value = "0.25"
    updateWeightDisplay.call(document.getElementById(`weight-${desc}`))
  })

  // Reset test image
  removeTestImage()

  // Hide results
  document.getElementById("results-panel").style.display = "none"

  showToast("Formulaire réinitialisé.", "info")
}

// Enhanced particle burst for AI theme
function createParticleBurst(x, y) {
  const burstParticles = []
  const colors = ["rgba(34, 197, 94, ", "rgba(59, 130, 246, ", "rgba(168, 85, 247, ", "rgba(249, 115, 22, "]

  for (let i = 0; i < 30; i++) {
    const angle = (Math.PI * 2 * i) / 30
    const velocity = Math.random() * 8 + 3
    const color = colors[Math.floor(Math.random() * colors.length)]

    burstParticles.push({
      x: x,
      y: y,
      vx: Math.cos(angle) * velocity,
      vy: Math.sin(angle) * velocity,
      life: 1,
      decay: Math.random() * 0.02 + 0.01,
      color: color,
      size: Math.random() * 4 + 2,
    })
  }

  function animateBurst() {
    burstParticles.forEach((particle, index) => {
      particle.x += particle.vx
      particle.y += particle.vy
      particle.life -= particle.decay
      particle.vx *= 0.98
      particle.vy *= 0.98

      if (particle.life <= 0) {
        burstParticles.splice(index, 1)
        return
      }

      ctx.beginPath()
      ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2)
      ctx.fillStyle = particle.color + particle.life + ")"
      ctx.fill()

      // Glow effect
      ctx.beginPath()
      ctx.arc(particle.x, particle.y, particle.size * 2, 0, Math.PI * 2)
      ctx.fillStyle = particle.color + particle.life * 0.3 + ")"
      ctx.fill()
    })

    if (burstParticles.length > 0) {
      requestAnimationFrame(animateBurst)
    }
  }

  animateBurst()
}

// Initialize the application
document.addEventListener("DOMContentLoaded", () => {
  initializeCanvas()
  createFloatingClusters()
  initializeEventListeners()
  startAnimation()
})

// Update weight display
function updateWeightDisplay() {
  const value = this.value
  const valueDisplay = this.parentElement.querySelector(".weight-value")
  valueDisplay.textContent = Number.parseFloat(value).toFixed(2)
}

// Close modal when clicking outside
document.getElementById("cluster-modal").addEventListener("click", function (event) {
  if (event.target === this) {
    closeModal()
  }
})

// Keyboard shortcuts
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeModal()
    closeConfig()
  }
})

// Toast notification system
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container")
  const toast = document.createElement("div")
  toast.className = `toast ${type}`
  toast.textContent = message

  container.appendChild(toast)

  setTimeout(() => {
    toast.remove()
  }, 4000)

  toast.addEventListener("click", () => {
    toast.remove()
  })
}
