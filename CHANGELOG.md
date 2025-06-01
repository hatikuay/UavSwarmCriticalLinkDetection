# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]
- Planned: Add unit tests for `uav_swarm.py`.
- Planned: Refactor channel-quality estimator to use GPU acceleration.

## [1.1.0] – 2025-05-20
### Added
- PyQt5 GUI in `simulation_pyQT.py`.  
- Real‐time plotting of λ₂ and risky links.  
- “Generate Report” button to force‐write `uav_report.txt`.

### Changed
- Updated default parameter sweep range for α, β, γ.  
- Improved logging format in `parametre.py`.

### Fixed
- Bug in partition counting when β < 0.15.

## [1.0.0] – 2025-05-10
### Added
- Initial `UAVState` and `UAVSwarm` classes in `uav_swarm.py`.  
- `simulation.py` console simulation script.  
- `parametre.py` for parameter sweep.  
- Documentation: `README.md`, `LICENSE`.  
