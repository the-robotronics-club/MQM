# Mess Queue Management System - Core Team Proposal

## 1. Project Overview & Objectives

**Project Duration:** Mid-August through September

**Goal:** Address the lack of real-time visibility into mess congestion and reduce peak-hour overcrowding.

**Solution:** A live dashboard displaying real-time queue size, seat occupancy, headcount, and crowd level to enable students to make informed dining decisions.

**Scope:** Deployment of two (2) sensing units to cover the full mess area.(though we will start with 1 initially)

## 2. System Architecture

**Edge Sensors (Perception Layer):** 2x Raspberry Pi Zero boards paired with Pi Camera modules. These will capture high-quality snapshots periodically (e.g., every 10 seconds).

**Enclosure & Power:** Housed in custom 3D-printed enclosures and powered via standard USB adapters, with power-backup considerations.

- **Data Transmission:** Snapshots are transmitted over the IIT Mandi Campus Wi-Fi to a central server with low latency.

**Central Processing (Server):** Receives images and runs YOLOv8 object detection inference to estimate crowd metrics and queue length. Data is stored in a Time-Series Database (e.g., InfluxDB).

**User Interface:** A Web Dashboard (React/Next.js) fetching data via REST APIs from the central server.

## 3. Team Allotment & Responsibilities

### Web Development & Computer Vision (Web-D + CV)

**Team Members:** Curio, Agam, Saiprasanth, Ariyan, Ishan

- **Computer Vision Pipeline:** Implement and optimize the YOLOv8 object detection inference on the central server for accurate headcount, seat occupancy, and queue length estimation.

- **Backend Server:** Set up the API Gateway (FastAPI/Node.js) and Time-Series Database (InfluxDB) to receive and log data from the Pi units.

- **Web Dashboard:** Develop the frontend (React/Next.js) to provide a seamless, live UI for students to check mess congestion.

### RPi Power Management (BMS, Backup)

#### **Team Member:** Ritisha

- **Power Delivery System:** Manage the deployment of standard USB adapters for the sensing units.

- **Backup & BMS:** Design and integrate a Battery Management System (BMS) or UPS backup to ensure continuous operation during short power fluctuations and enable safe shutdowns.

- **Power Optimization:** Monitor and validate the thermal and power draw of the edge units running 24/7.(see what we can do about thermal power dissipation though not that imp for now)

### RPi OS & Programming

#### **Team Member:** Kartik (Lead)

- **System Configuration:** Flash and configure the Raspberry Pi Zero OS for headless, stable operation on the campus Wi-Fi network.

- **Capture & Transmission Scripts:** Write lightweight Python scripts utilizing `libcamera` to wake up, capture JPEGs, and reliably publish them via HTTP POST or MQTT to the central server every 10 seconds.

- **Edge Reliability:** Implement watchdogs and auto-restart mechanisms to ensure the edge scripts recover from network drops.

### Mechanical

**Team Members:** Suman, Priyanka

- **Enclosure Design:** CAD design and 3D print custom enclosures to house the Pi Zero, Camera, and power modules securely.

- **Thermal & Mounting:** Ensure the enclosure allows for passive cooling and provides the correct mounting angles for optimal camera Field of View (FOV).

- **Physical Deployment:** Securely mount the 2 sensing units at the identified vantage points in the mess hall.


## Infra & Local Setup

### 1. Environment Variables
Copy `.env.example` to `.env` and fill in your credentials:
```bash
cp .env.example .env