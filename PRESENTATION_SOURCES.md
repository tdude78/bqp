# Sources for the BQP space domain awareness presentation

This is the reference list for the presentation delivered to BQP, compiled by
auditing the deck source (`presentation.tex`, 32 pages: title, 13 numbered
content slides, a closing statement, a backup divider, and 16 discussion
backups) together with the research notes behind it.

Part 1 lists the 47 sources the deck actually cites, with the slides that use
each one, the claim it supports, and the qualification that travels with that
claim. Part 2 lists a further 204 sources consulted while researching the deck
but not cited on a slide.

Everything here is third-party public material. The links point to the
publishers, who own the works. Nothing in this file is a BQP performance claim
or a claim about the author's own work.

Compiled 2026-09-03. Sources were last verified live on 2026-08-31 during the
deck closeout; see "Verification record" at the end.

## How to read the entries

The deck labels every claim with the kind of evidence behind it, and those
labels carry over here:

- **Primary** is an official agency, standards body, product owner, or
  first-party technical document.
- **Company** means BQP or another vendor reporting on its own work.
- **Study** is a peer-reviewed paper or a conference technical paper.
- **Preprint** is a paper that has not completed peer review.
- **Secondary** is a credible report that does not supply the underlying record.
- **Derived** is the presenter's own analysis, proposal, or arithmetic, and is
  not sourced to anyone else.

A claim marked as a study result is never a claim about BQP's performance.
Several entries below carry an explicit qualification for exactly that reason.

---

# Part 1: cited in the deck

## Primary and official

**ESA DISCOS statistics.** <https://sdup.esoc.esa.int/discosweb/statistics/>  
*Slides:* The mission pressure is measurable.  
*Supports:* about 46,590 objects regularly tracked in Earth orbit as of 31 July 2026.  
*Qualification:* date-sensitive, and the tracked count is not the same population as the debris estimate below.

**ESA Space Environment Report 2026.** <https://www.sdo.esoc.esa.int/environment_report/Space_Environment_Report_latest.pdf>  
*Slides:* The mission pressure is measurable.  
*Supports:* roughly 1.2 million debris objects between 1 and 10 cm; the catalogued-object history in Figure 2.1 exceeding 40,000 objects; and, in backup, a business-as-usual 200-year risk index about four times the reference sustainability threshold.  
*Qualification:* the Figure 2.1 series is limited by surveillance capability and uses ESA object classes. It is neither the DISCOS tracked count nor the 1 to 10 cm estimate. The risk index is a projection, not measured current risk. Reuse of the figure is personal and non-commercial with source and issue-date credit.

**Mashiku et al., NASA CARA AI/ML assessment, AMOS 2025.** <https://ntrs.nasa.gov/api/citations/20250008251/downloads/AMOS_2025_AIML_Paper_UpdatedContractorAddress.pdf>  
*Slides:* Trust is a measured operating property; Backup: benchmark protocol.  
*Supports:* operational trust depends on state accuracy, covariance realism, screening behaviour, and safe handling outside the training distribution. The reviewed LSTM example had 27 high-risk events among 782 total, with 211 m average and 2,480 m maximum miss-distance prediction error at an average 1.8 days to time of closest approach. The paper concludes that fully automated AI/ML operational risk assessment is not currently viable and favours hybrid approaches plus covariance realism.  
*Qualification:* study data, not BQP performance. The architecture the deck proposes is informed by CARA, not prescribed by NASA.

**NASA Conjunction Assessment Handbook, Revision 2, Volume 2.** <https://www.nasa.gov/wp-content/uploads/2026/08/ca-handbook-volume2-nasa-sp-20205011318-rev2-vol2.pdf>  
*Slides:* Sources and evidence labels (referenced from the trust and benchmark backups).  
*Supports:* covariance-realism and probability-of-collision methods that can anchor benchmark concordance.  
*Qualification:* apply the exact NASA method and assumptions only where the customer mission and data support them. Being a benchmark reference is not certification.

**NASA robotic CARA probability-of-collision methods.** <https://ntrs.nasa.gov/api/citations/20190011726/downloads/20190011726.pdf>  
*Slides:* Backup: screening economics, and the miss budget.  
*Supports:* the Foster, Chan, Patera and Alfano probability-of-collision methods, with CARA using Foster.  
*Qualification:* primary method reference only.

**Office of Space Commerce, TraCSS conjunction-assessment verification dataset.** <https://space.commerce.gov/tracss-publishes-dataset-for-conjunction-assessment-verification/>  
*Slides:* Year 1: productize and prove; Backup: benchmark protocol; Backup: the benchmark ladder.  
*Supports:* a public, CC0 verification set the product would be measured against.  
*Qualification:* BQP had not run this set at the time of the deck.

**Office of Space Commerce, TraCSS programme.** <https://space.commerce.gov/traffic-coordination-system-for-space-tracss/>  
*Slides:* How BQP wins in a market that already exists; Backup: competitor evidence changes the wedge.  
*Supports:* the civil traffic-coordination programme as a market and integration route.

**TraCSS CDM and OCM field recommendations, January 2026.** <https://www.space.commerce.gov/wp-content/uploads/Recommendation-on-TraCSS-CDM-Fields.pdf>  
*Slides:* Reference architecture: integrate, do not displace; Backup: integration and acquisition routes.  
*Supports:* the civil route has published message interfaces to build against.

**ESA Kelvins collision-avoidance challenge dataset.** <https://kelvins.esa.int/collision-avoidance-challenge/data/>  
*Slides:* Year 1: productize and prove; Backup: benchmark protocol; Backup: the benchmark ladder.  
*Supports:* a public CDM dataset for benchmarking.  
*Qualification:* BQP had not run this set at the time of the deck.

**Spaceflight Safety Handbook for Satellite Operators, version 1.7.** <https://www.space-track.org/documents/SFS_Handbook_For_Operators_V1.7.pdf>  
*Slides:* Sources and evidence labels.  
*Supports:* the operator-facing conjunction-assessment process as practised today.

**NASA Starling.** <https://www.nasa.gov/smallspacecraft/what-is-starling/>  
*Slides:* Year 3: qualify bounded autonomy; One engine, three placements; Backup: edge autonomy exists, qualification is the wedge; Backup: self-hosted means bounded model placement.  
*Supports:* demonstrated autonomous spacecraft coordination and a simulated autonomous conjunction-mitigation workflow; flight evidence for onboard estimation and planning.  
*Qualification:* an existence proof, not BQP heritage.

**NASA Starling onboard navigation update, 17 August 2026.** <https://www.nasa.gov/blogs/smallsatellites/2026/08/17/nasas-starling-mission-opens-new-frontiers-in-space-navigation/>  
*Slides:* Year 3: qualify bounded autonomy; Backup: edge autonomy exists, qualification is the wedge.  
*Supports:* a navigation experiment refined the orbits of more than 200 space objects onboard, without ground intervention, over three days.  
*Qualification:* an existence proof, not proof of calibrated onboard BQPhy propagation.

**NASA/TP-20260005089, Multi-Agent Swarm State of the Art Report, 2026.** <https://ntrs.nasa.gov/citations/20260005089>  
*Slides:* Backup: constellation maintenance is a constrained loop.  
*Supports:* Starling ROMEO combined navigation, orbital-maintenance planning, and maneuver-command services; generated maneuvers still received ground conjunction screening and a final go/no-go; explicit delta-v limits applied; and the demonstration was only partly successful.  
*Qualification:* an external operational lesson supporting staged authority and independent screening. It is not proof of general autonomous constellation maintenance, and it is not BQP heritage.

**NASA formal runtime assurance.** <https://ntrs.nasa.gov/citations/20240006522>  
*Slides:* Backup: authority follows assurance; Backup: SOTA bets and kill gates.  
*Supports:* runtime assurance can monitor an advanced component and switch to a trusted reversionary component when a safety property is threatened.  
*Qualification:* an assurance architecture reference. Not a space collision-avoidance certification, and not a claim that BQPhy implements Simplex runtime assurance today.

**ASTM F3269-21, run-time assurance practice.** <https://store.astm.org/f3269-21.html>  
*Slides:* Backup: authority follows assurance; Backup: SOTA bets and kill gates.  
*Supports:* the structure of the proposed authority ladder.  
*Qualification:* the practice is written for aircraft, not space certification. The deck notes that DoD Directive 3000.09 is not a blanket rule for non-weapon collision avoidance.

**ONNX Runtime execution providers.** <https://onnxruntime.ai/docs/execution-providers/>  
*Slides:* One engine, three placements; Backup: self-hosted means bounded model placement; Backup: the edge acceptance sheet.  
*Supports:* one inference interface across CPU, GPU, FPGA, and specialised accelerator providers.  
*Qualification:* a portability mechanism only, and the documentation itself separates production from preview providers. Operator coverage, compiled artifacts, dependencies, precision, and performance still need target-specific validation.

**NASA High Performance Spaceflight Computing.** <https://www.nasa.gov/game-changing-development-projects/high-performance-spaceflight-computing-hpsc/>  
*Slides:* One engine, three placements; Backup: self-hosted means bounded model placement; Backup: the edge acceptance sheet.  
*Supports:* higher-performance, power-managed, fault-tolerant spaceflight computing that supports AI/ML and autonomy.  
*Qualification:* enabling-compute evidence. It does not make HPSC qualified or available for a BQPhy mission without integration and test, and Jetson-class execution is not flight qualification.

**CCSDS 502.0-B-3, Orbit Data Messages.** <https://ccsds.org/Pubs/502x0b3e1.pdf>  
*Slides:* Reference architecture: integrate, do not displace.  
*Supports:* OMM and OEM as the appropriate common exchange products for orbit data.  
*Qualification:* exact message and field compliance must be defined during product design.

**CCSDS 508.0-B-1, Conjunction Data Message.** <https://ccsds.org/Pubs/508x0b1e2c2.pdf>  
*Slides:* Reference architecture: integrate, do not displace.  
*Supports:* the CDM as the common conjunction exchange product.  
*Qualification:* as above.

**Space Systems Command, Unified Data Library.** <https://www.ssc.spaceforce.mil/Newsroom/Article/4039108/space-systems-commands-udl-provides-data-solutions-at-the-speed-of-battle>  
*Slides:* Reference architecture: integrate, do not displace.  
*Supports:* the UDL as a government data and integration route for operational space data.  
*Qualification:* a valuable interface target, not a prerequisite for every government sale. This page returned HTTP 403 to an automated client during the link check, which reflects site access control rather than a dead page.

## Studies and preprints

**Acciarini, G., Baydin, A. G., and Izzo, D. (2024). "Closing the Gap Between SGP4 and High-Precision Propagation via Differentiable Programming" (dSGP4 and ML-dSGP4).** arXiv:2402.04830, <https://arxiv.org/abs/2402.04830>. Published in Acta Astronautica 226(1) (2025) 8, DOI 10.1016/j.actaastro.2024.10.063.  
*Slides:* Backup: SOTA bets and kill gates.  
*Supports:* SGP4 made differentiable, with learned input and output corrections that behave as identity operators by default.  
*Qualification:* an open architecture and benchmark reference, never BQP heritage.

**Priestley, C., and Handley, W. (2026). "jaxsgp4: GPU-accelerated mega-constellation propagation with batch parallelism".** arXiv:2603.27830, submitted 29 March 2026, <https://arxiv.org/abs/2603.27830>.  
*Slides:* Raw propagation speed is commoditizing (Figure 1 reproduced in backup).  
*Supports:* 9,341 satellites propagated to 1,000 future times in 3.8 ms on a single A100, and about 1,500x maximum speedup over the paper's own C++ baseline.  
*Qualification:* large-batch JAX GPU FP32 against C++ CPU FP64, under the paper's declared comparison. It is a study result, not BQP performance, and it establishes no accuracy, covariance realism, or safety claim. The deck uses it to demote raw speed as a moat.

**Parker, W. E., and Linares, R. (2024). "Satellite Drag Analysis During the May 2024 Gannon Geomagnetic Storm".** Journal of Spacecraft and Rockets 61(5), 1412-1416. arXiv:2406.08617, <https://arxiv.org/abs/2406.08617>.  
*Slides:* Raw propagation speed is commoditizing; Backup: where the LEO error actually lives.  
*Supports:* a two to four times degradation during the storm, with the geomagnetic ap forecast poor even one day ahead.  
*Qualification:* one storm, TLE-derived. Drag is not the only error source.

**Wang, R., and Bai, X. (2026). "A Machine-Learning-Based Global Thermospheric Density Forecasting Model" (AETHER-P3).** arXiv:2608.00352, submitted 31 July 2026, <https://arxiv.org/abs/2608.00352>. Journal version in Space Weather 24(6), DOI 10.1029/2026SW004968.  
*Slides:* Backup: where the LEO error actually lives; Backup: SOTA bets and kill gates.  
*Supports:* learned thermospheric density with calibrated uncertainty over roughly 300 to 520 km.  
*Qualification:* a preprint, and not BQP work. The storm mode the deck proposes is a proposal.

**Moody, A., Axelrad, P., and Russell, R. (2026). "Machine Learning Argument of Latitude Error Model for LEO Satellite Orbit and Covariance Correction".** arXiv:2602.16764, <https://arxiv.org/abs/2602.16764>. In the 2026 IEEE Aerospace Conference.  
*Slides:* Backup: SOTA bets and kill gates.  
*Supports:* a correction acting along the dominant error dimension.  
*Qualification:* research direction, not BQP heritage.

**Shahid, M. B., Jiang, Z., Sarkar, S., and Fleming, C. (2026). "Continuous-Time Probabilistic Correctors for Uncertainty-Aware Physics-Based Spacecraft Trajectory Forecasting".** arXiv:2606.21021, submitted 19 June 2026, <https://arxiv.org/abs/2606.21021>.  
*Slides:* Sources and evidence labels (referenced from the SOTA backup).  
*Supports:* a physics predictor with a learned continuous-time corrector that emits full-covariance, heavy-tailed (Student-t) predictive uncertainty, evaluated with proper scoring and Mahalanobis calibration.  
*Qualification:* preprint status and single-study scope. A current research direction only, labelled as a preprint on the slide and in the source map.

**Stevenson, E., Rodriguez-Fernandez, V., Urrutxua, H., and Camacho, D. (2023). "Benchmarking deep learning approaches for all-vs-all conjunction screening".** Advances in Space Research 72(7), 2660-2675, DOI 10.1016/j.asr.2023.01.036. Open-access copy: <https://oa.upm.es/80898/1/10020279.pdf>.  
*Slides:* Backup: screening economics, and the miss budget.  
*Supports:* 170 million pairs screened over 7 days in the CNES BAS3E simulation.  
*Qualification:* a study result, not BQP performance.

**Holzinger, M. J., Scheeres, D. J., and Alfriend, K. T. (2012). "Object Correlation, Maneuver Detection, and Characterization Using Control Distance Metrics".** Journal of Guidance, Control, and Dynamics 35(4), 1312-1325, DOI 10.2514/1.53245, <https://arc.aiaa.org/doi/10.2514/1.53245>.  
*Slides:* Backup: SOTA bets and kill gates.  
*Supports:* a principled control-effort distance for associating tracks across a maneuver.  
*Qualification:* method reference.

**Park, I., Stevenson, M., Nicolls, M., et al. (LeoLabs, 6 authors) (2019). "Statistical Covariance Realism Assessment of LeoLabs' Orbit Determination System".** AMOS 2019, <https://amostech.com/TechnicalPapers/2019/Astrodynamics/Park.pdf>.  
*Slides:* Trust is a measured operating property.  
*Supports:* covariance consistency as a measurable acceptance property.  
*Qualification:* method and assessment reference.

**Escribano, G., Sanjurjo-Rivo, M., Siminski, J., Pastor, A., and Escobar, D. (2021). "Automatic maneuver detection and tracking of space objects in optical survey scenarios based on stochastic hybrid systems formulation".** arXiv:2109.07801, <https://arxiv.org/abs/2109.07801>.  
*Slides:* Sources and evidence labels (referenced from the SOTA backup).  
*Supports:* a stochastic-hybrid formulation with sequential Monte Carlo filtering that jointly detects and tracks maneuvering space objects online.  
*Qualification:* a research method, not BQP heritage or proof of performance on BQP data. One candidate to compare against innovation tests and customer incumbents.

**Hofer et al., maneuver custody through hybrid track association, AMOS 2024.** <https://amostech.com/TechnicalPapers/2024/SDA/Hofer.pdf>  
*Slides:* Sources and evidence labels (referenced from the SOTA backup).  
*Supports:* hybrid kinematic and non-kinematic track association as a path to maintaining custody through maneuvers.  
*Qualification:* the strongest quoted precision and recall include simulated maneuver-related tracks. The deck does not quote them as operational performance.

**Ferrara, F., Schillinger Arana, L. W., Dörfler, F., and Li, S. H. Q. (2025). "A Markov Decision Process Framework for Early Maneuver Decisions in Satellite Collision Avoidance".** arXiv:2508.05876, v1 7 August 2025, v2 10 December 2025, <https://arxiv.org/abs/2508.05876>.  
*Slides:* Backup: constellation maintenance is a constrained loop.  
*Supports:* commit timing trading propellant against residual risk.  
*Qualification:* synthetic events. External evidence, not BQP heritage.

**Wang, H., and Ning, C. (2025). "Conformal Prediction in The Loop: A Feedback-Based Uncertainty Model for Trajectory Optimization".** arXiv:2510.16376, <https://arxiv.org/abs/2510.16376>. Accepted to the NeurIPS 2025 main track.  
*Slides:* Trust is a measured operating property.  
*Supports:* constructing uncertainty sets with explicit coverage guarantees.  
*Qualification:* method evidence for a proposed coverage commitment. Never current BQP performance.

**Feldman, A. O., Harp, D. I., Duncan, J., et al. (4 authors) (2025). "Conformal Safety Monitoring for Flight Testing: A Case Study in Data-Driven Safety Learning".** arXiv:2511.20811, <https://arxiv.org/abs/2511.20811>. ICRA 2025 workshop.  
*Slides:* Trust is a measured operating property.  
*Supports:* calibrating a flight-safety classifier with conformal prediction, with matching theoretical guarantees.  
*Qualification:* as above.

**Antunes de Sá, A., Shouppe, M., Takahashi, S., et al. (7 authors) (2023). "Characterizing a Novel Coordinated Optimal Avoidance Maneuver Framework for Space Traffic Management".** AMOS 2023 poster session, <https://amostech.com/TechnicalPapers/2023/Poster/Antunes_de_Sa.pdf>.  
*Slides:* How BQP wins in a market that already exists; Backup: competitor evidence changes the wedge.  
*Supports:* Kayhan unveiling a machine-to-machine interface for autonomous coordination and pre-coordination of maneuver responsibility.  
*Qualification:* durable company-technical evidence about a competitor. The deck attaches no catalogue-size language to it.

**Kim, S., Ahn, S.-W., Suh, I.-S., et al. (6 authors) (2025). "Quantum annealing for combinatorial optimization: a benchmarking study".** npj Quantum Information 11(1), article 77, DOI 10.1038/s41534-025-01020-1, <https://www.nature.com/articles/s41534-025-01020-1>.  
*Slides:* Backup: quantum language that survives diligence.  
*Supports:* the evidence bar a quantum speedup claim has to clear.  
*Qualification:* used to set the bar, not to claim a BQP result.

**Quinton, F. A., Myhr, P. A. S., Barani, M., et al. (5 authors) (2025). "Quantum annealing applications, challenges and limitations for optimisation problems compared to classical solvers".** Scientific Reports 15(1), article 12733, DOI 10.1038/s41598-025-96220-2, <https://www.nature.com/articles/s41598-025-96220-2>.  
*Slides:* Backup: quantum language that survives diligence.  
*Supports:* the same evidence bar.  
*Qualification:* as above.

## Company and product pages

**BQP, first federal contract for quantum-assisted AI for space domain awareness (PC-QAML).** <https://www.bqpsim.com/press/bqp-awarded-first-federal-contract-to-advance-quantum-assisted-ai-for-space-domain-awareness>  
*Slides:* Year 1: productize and prove; Backup: capability maturity; Backup: self-hosted means bounded model placement; Backup: the edge acceptance sheet.  
*Supports:* BQP reports that the SpaceWERX PC-QAML effort will develop and validate physics-constrained, quantum-assisted ML for space object identification and behaviour characterization. In backup, BQP reports a 14 million to 2,000 parameter reduction, better than 99 percent classification accuracy, up to 10 times lower inference latency, about 90 percent lower power, and a Jetson Nano demonstration.  
*Qualification:* company-reported, and future work under contract. Task, dataset, comparator, and measurement conditions are not fully disclosed. This is compact-classification evidence. It does not establish propagation, orbit determination, uncertainty quantification, control, or flight performance, and it is not an independently validated operational product.

**BQP product overview.** <https://www.bqpsim.com/product-overview>  
*Slides:* Backup: quantum language that survives diligence.  
*Supports:* commercially available quantum-inspired optimization executing on current HPC.  
*Qualification:* company-reported. Physics-based and data-driven solvers are labelled R&D on the page. The deck infers no CPU or GPU implementation detail and does not promote the R&D claims.

**Starlink Space Safety.** <https://space-safety.starlink.com/>  
*Slides:* How BQP wins in a market that already exists.  
*Supports:* operator conjunction screening in under one minute, supporting covariance-bearing trajectories and CDMs.  
*Qualification:* this is a strong basic-screening baseline. The deck does not call it uncalibrated or share-to-play.

**Starlink CDM documentation.** <https://docs.space-safety.starlink.com/docs/tutorial-basics/cdms/>  
*Slides:* Backup: competitor evidence changes the wedge.  
*Supports:* the CDM interface behind the screening service above.

**COMSPOC SSASuite.** <https://www.comspoc.com/ssasuite>  
*Slides:* How BQP wins in a market that already exists; Backup: competitor evidence changes the wedge.  
*Supports:* established products already combine propagation, uncertainty, conjunction analysis, and maneuver characterization.  
*Qualification:* qualitative market positioning. BQP cannot claim nobody sells these functions.

**Slingshot Aerospace Processing Suite.** <https://www.slingshot.space/product-processing-suite>  
*Slides:* as above. Same claim and qualification.  

**LeoLabs conjunction alerts.** <https://leolabs.space/conjunction-alerts/>  
*Slides:* as above. Same claim and qualification.  

**GMV Focusoc.** <https://www.gmv.com/en/products/space/focusoc>  
*Slides:* as above. Same claim and qualification.  

## Programme and route pages

**BMC3I TAP Lab 26-A cohort applications.** <https://bmc3i.catalystcampus.org/26-a-cohort-applications-page>  
*Slides:* Backup: integration and acquisition routes.  
*Supports:* the 26-A cohort was closed when checked on 28 August 2026.  
*Qualification:* date-sensitive.

**SpaceWERX funding opportunities.** <https://spacewerx.us/get-funded/>  
*Slides:* Backup: integration and acquisition routes.  
*Supports:* live funding routes.  
*Qualification:* the routes shown in the deck are examples, and the page changes.

---

# Part 2: background research corpus

These 204 sources were consulted while building the deck and are recorded in
the research notes, but no slide cites them. They are listed by topic with the
publisher's own title or identifier. No claim in the deck rests on them.

### Orbit propagation state of the art

- [AIAA 10.2514/1.G008245](https://arc.aiaa.org/doi/10.2514/1.G008245)
- [AMOS 2025 technical paper, Badura](https://amostech.com/TechnicalPapers/2025/Machine-Learning-for-SDA-Applications/Badura.pdf)
- [arXiv:2207.08993](https://arxiv.org/abs/2207.08993)
- [arXiv:2210.08364](https://arxiv.org/abs/2210.08364)
- [arXiv:2210.16992](https://arxiv.org/abs/2210.16992)
- [arXiv:2405.19384](https://arxiv.org/abs/2405.19384)
- [arXiv:2506.12007](https://arxiv.org/abs/2506.12007)
- [arXiv:2509.08607](https://arxiv.org/abs/2509.08607)
- [arXiv:2511.06105](https://arxiv.org/abs/2511.06105)
- [arXiv:2605.16078](https://arxiv.org/abs/2605.16078)
- [arXiv:2606.30936](https://arxiv.org/abs/2606.30936)
- [bluescarni.github.io](https://bluescarni.github.io/heyoka.py/notebooks/sgp4_propagator.html)
- [DOI 10.2514/6.2024-1862](https://doi.org/10.2514/6.2024-1862)
- [GitHub esa/dSGP4](https://github.com/esa/dSGP4)
- [GitHub spaceml-org/karman](https://github.com/spaceml-org/karman)
- [SINDyc reduced-order assimilation, arXiv 2604.24646](https://arxiv.org/abs/2604.24646)

### Uncertainty quantification and calibration

- [AIAA 10.2514/1.57599](https://arc.aiaa.org/doi/10.2514/1.57599)
- [arXiv:2406.02436](https://arxiv.org/abs/2406.02436)
- [Covariance determination for uncertainty realism](https://www.sciencedirect.com/science/article/pii/S0273117722007190)
- [researchgate.net](https://www.researchgate.net/publication/299346568)

### Conjunction screening and assessment

- [AMOS 2023 technical paper, Hejduk](https://amostech.com/TechnicalPapers/2023/Conjunction-RPO/Hejduk.pdf)
- [arXiv:2002.00430](https://arxiv.org/abs/2002.00430)
- [arXiv:2410.20928](https://arxiv.org/abs/2410.20928)
- [CAM design methods review, arXiv 2503.22555](https://arxiv.org/abs/2503.22555)
- [Cara faq npr handbook 2025.pdf (nasa.gov)](https://www.nasa.gov/wp-content/uploads/2025/03/cara-faq-npr-handbook-2025.pdf)
- [conference.sdo.esoc.esa.int](https://conference.sdo.esoc.esa.int/proceedings/sdc9/paper/209/SDC9-paper209.pdf)
- [esa.int](https://www.esa.int/Space_Safety/Space_Debris/ESA_Space_Environment_Report_2025)
- [esa.int](https://www.esa.int/Space_Safety/Space_Debris/CREAM_avoiding_collisions_in_space_through_automation)
- [Kelvins challenge results paper, arXiv 2008.03069](https://arxiv.org/abs/2008.03069)
- [NASA multistep Pc algorithm, 2023](https://ntrs.nasa.gov/api/citations/20230010175/downloads/Hall_Baars_Casali_ASC_2023_08_13_PcMultiStep_Paper.pdf)
- [NASA NTRS 20190028900](https://ntrs.nasa.gov/api/citations/20190028900/downloads/20190028900.pdf)
- [S42064 021 0125 x (link.springer.com)](https://link.springer.com/article/10.1007/s42064-021-0125-x)
- [Senate appropriators retain funding for noaas tracss space traffic system (spacepolicyonline.com)](https://spacepolicyonline.com/news/senate-appropriators-retain-funding-for-noaas-tracss-space-traffic-system/)
- [Space.com](https://www.space.com/space-exploration/satellites/every-spacex-starlink-satellite-has-to-dodge-a-collision-almost-weekly-and-experts-fear-the-worst)
- [Tracss implements program increment 1 2 on demand ephemerides screening and bulk submissions now live (space.commerce.gov)](https://space.commerce.gov/tracss-implements-program-increment-1-2-on-demand-ephemerides-screening-and-bulk-submissions-now-live/)

### Maneuver detection and behaviour characterization

- [afresearchlab.com](https://afresearchlab.com/technology/oracle/)
- [Afrl awards oracle contract (advancedspace.com)](https://advancedspace.com/afrl-awards-oracle-contract/)
- [AIAA 10.2514/6.2025-98101](https://arc.aiaa.org/doi/10.2514/6.2025-98101)
- [arXiv:2507.08234](https://arxiv.org/abs/2507.08234)
- [Bayesian low-thrust tracking, arXiv 2410.18300](https://arxiv.org/abs/2410.18300)
- [China demonstrated satellite dogfighting space force general says (defensenews.com)](https://www.defensenews.com/space/2025/03/18/china-demonstrated-satellite-dogfighting-space-force-general-says/)
- [Chinese spacecraft prepare for orbital refueling test as us surveillance sats lurk nearby (spacenews.com)](https://spacenews.com/chinese-spacecraft-prepare-for-orbital-refueling-test-as-us-surveillance-sats-lurk-nearby/)
- [Continuously maneuvering Starlink OD, unscented batch filter](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12252113/)
- [Dancing lights space how manage risks satellite close approaches geostationary orbit (csis.org)](https://www.csis.org/analysis/dancing-lights-space-how-manage-risks-satellite-close-approaches-geostationary-orbit)
- [Dynamic space operations adapting aerospaces capabilities support contested space domain (aerospace.org)](https://aerospace.org/article/dynamic-space-operations-adapting-aerospaces-capabilities-support-contested-space-domain)
- [eoportal.org](https://www.eoportal.org/satellite-missions/shijian-21)
- [Exoanalytic launches exotrack an advanced service for satellite operations in geo (prnewswire.com)](https://www.prnewswire.com/news-releases/exoanalytic-launches-exotrack-an-advanced-service-for-satellite-operations-in-geo-302093214.html)
- [exoanalytic.com](https://exoanalytic.com/space-intelligence/exotrack/)
- [GEO electric propulsion passive OD](https://www.sciencedirect.com/science/article/abs/pii/S2468896724000934)
- [How russia is intercepting communications from european (rand.org)](https://www.rand.org/pubs/commentary/2026/03/how-russia-is-intercepting-communications-from-european.html)
- [keeptrack.space](https://keeptrack.space/deep-dive/leolabs)
- [Leolabs announces next generation expeditionary radar for advanced space domain awareness missions (leolabs.space)](https://leolabs.space/press/leolabs-announces-next-generation-expeditionary-radar-for-advanced-space-domain-awareness-missions/)
- [Nga space force ink accord on responsibilities for buying commercial isr (breakingdefense.com)](https://breakingdefense.com/2025/05/nga-space-force-ink-accord-on-responsibilities-for-buying-commercial-isr/)
- [Reachability-based search for maneuvering satellites](https://link.springer.com/article/10.1007/s40295-023-00365-z)
- [Russian luch olymp 2 satellite approaching multiple geo spacecraft (slingshot.space)](https://www.slingshot.space/news/russian-luch-olymp-2-satellite-approaching-multiple-geo-spacecraft)
- [Russias new cosmos satellite orbiting near us sat piques asat fears (breakingdefense.com)](https://breakingdefense.com/2025/05/russias-new-cosmos-satellite-orbiting-near-us-sat-piques-asat-fears/)
- [S40295 022 00311 5 (link.springer.com)](https://link.springer.com/article/10.1007/s40295-022-00311-5)
- [S40295 024 00478 z (link.springer.com)](https://link.springer.com/article/10.1007/s40295-024-00478-z)
- [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0094576523004800)
- [sciencedirect.com](https://www.sciencedirect.com/science/article/abs/pii/S0273117724013048)
- [sdataplab.org](https://sdataplab.org/)
- [Second russian luch olymp satellite now trailing western systems in orbit (breakingdefense.com)](https://breakingdefense.com/2023/10/second-russian-luch-olymp-satellite-now-trailing-western-systems-in-orbit/)
- [Slingshot unveils ai that spots satellite anomalies and potential bad actors (spacenews.com)](https://spacenews.com/slingshot-unveils-ai-that-spots-satellite-anomalies-and-potential-bad-actors/)
- [Space force battle management c2 tools kronos (airandspaceforces.com)](https://www.airandspaceforces.com/space-force-battle-management-c2-tools-kronos/)
- [Space force contract andromeda program vendors (defensescoop.com)](https://defensescoop.com/2026/04/10/space-force-contract-andromeda-program-vendors/)
- [ssc.spaceforce.mil](https://www.ssc.spaceforce.mil/Portals/3/TAP%20Lab%20Project%20Apollo.pdf)
- [Starlink maneuver totals coverage](https://civl.com/news/story/starlink-satellites-execute-over-355-000-avoidance-maneuvers-in-one-year-as-orbi-c548991a)
- [ui.adsabs.harvard.edu](https://ui.adsabs.harvard.edu/abs/2010amos.confE..26H/abstract)
- [Unusual behavior in geo olymp k (aerospace.csis.org)](https://aerospace.csis.org/data/unusual-behavior-in-geo-olymp-k/)

### Edge computing and onboard autonomy

- [Ai on the edge of space (cset.georgetown.edu)](https://cset.georgetown.edu/publication/ai-on-the-edge-of-space/)
- [arXiv:2403.16677](https://arxiv.org/abs/2403.16677)
- [arXiv:2501.02667](https://arxiv.org/abs/2501.02667)
- [arXiv:2504.03983](https://arxiv.org/abs/2504.03983)
- [digitalcommons.usu.edu](https://digitalcommons.usu.edu/smallsat/2026/all2026/49/)
- [First blue ring mission to demonstrate space domain awareness with scout space sensor (blueorigin.com)](https://www.blueorigin.com/news/first-blue-ring-mission-to-demonstrate-space-domain-awareness-with-scout-space-sensor)
- [nvidianews.nvidia.com](https://nvidianews.nvidia.com/news/space-computing)
- [Sda issues broad agency announcement for pwsa battle management mission applications (sda.mil)](https://www.sda.mil/sda-issues-broad-agency-announcement-for-pwsa-battle-management-mission-applications/)
- [sda.mil](https://www.sda.mil/battle-management/)
- [Space operations in intelligence resilience and sustainability (aerospaceamerica.aiaa.org)](https://aerospaceamerica.aiaa.org/year-in-review/space-operations-in-2025-intelligence-resilience-and-sustainability/)

### Quantum and quantum-inspired computing

- [arXiv:2302.07181](https://arxiv.org/abs/2302.07181)
- [arXiv:2307.14419](https://arxiv.org/abs/2307.14419)
- [arXiv:2404.05516](https://arxiv.org/abs/2404.05516)
- [arXiv:2603.00701](https://arxiv.org/abs/2603.00701)
- [arXiv:2604.14287](https://arxiv.org/abs/2604.14287)
- [arXiv:2607.14150](https://arxiv.org/abs/2607.14150)
- [BQP quantum optimization for space blog](https://www.bqpsim.com/blogs/quantum-optimization-space)
- [bqpsim.com](https://www.bqpsim.com/)
- [Die dlr quantencomputing initiative (web.archive.org)](http://web.archive.org/web/20230328151736/https://qci.dlr.de/oekosystem/die-dlr-quantencomputing-initiative/)
- [DSN scheduling with quantum annealing, IEEE](https://ieeexplore.ieee.org/document/9863923/)
- [EO constellation scheduling on a D-Wave annealer](https://link.springer.com/chapter/10.1007/978-3-031-77432-4_16)
- [Esas first quantum computer will shift computing frontiers in space (philab.esa.int)](https://philab.esa.int/esas-first-quantum-computer-will-shift-computing-frontiers-in-space/)
- [Ionq commissions ground breaking quantum system at the u s air force (ionq.com)](https://www.ionq.com/news/ionq-commissions-ground-breaking-quantum-system-at-the-u-s-air-force)
- [Light curve classification scoping review](https://doi.org/10.3390/aerospace13030287)
- [Quantum annealing survey, arXiv 2602.03101](https://arxiv.org/abs/2602.03101)
- [Redefining aircraft navigation in a gps challenged world with airbus (q-ctrl.com)](https://q-ctrl.com/blog/redefining-aircraft-navigation-in-a-gps-challenged-world-with-airbus)
- [S40507 025 00369 8 (link.springer.com)](https://link.springer.com/article/10.1140/epjqt/s40507-025-00369-8)
- [S40507 025 00409 3 (epjquantumtechnology.springeropen.com)](https://epjquantumtechnology.springeropen.com/articles/10.1140/epjqt/s40507-025-00409-3)
- [S42005 024 01623 8 (nature.com)](https://www.nature.com/articles/s42005-024-01623-8)
- [S42064 024 0216 6 (link.springer.com)](https://link.springer.com/article/10.1007/s42064-024-0216-6)
- [trid.trb.org](https://trid.trb.org/View/2712112)

### Standards and interfaces

- [ASTM F3269 overview paper, AIAA SciTech 2021](https://arc.aiaa.org/doi/10.2514/6.2021-0525)

### SDA market landscape and sizing

- [Appropriators restore funding for commerces tracss spacewatch effort (breakingdefense.com)](https://breakingdefense.com/2025/07/appropriators-restore-funding-for-commerces-tracss-spacewatch-effort/)
- [Commerce extends commercial data contracts for space tracking system pilot (breakingdefense.com)](https://breakingdefense.com/2024/05/commerce-extends-commercial-data-contracts-for-space-tracking-system-pilot/)
- [Commercial price list (exoanalytic.com)](https://exoanalytic.com/commercial-price-list/)
- [Domain awareness counterspace systems top space force budget needs (defensenews.com)](https://www.defensenews.com/space/2024/09/17/domain-awareness-counterspace-systems-top-space-force-budget-needs/)
- [Global ssa market to reach 61b as governments prioritize space security resilience and orbital safety (nova.space)](https://nova.space/press-release/global-ssa-market-to-reach-61b-as-governments-prioritize-space-security-resilience-and-orbital-safety/)
- [Ground truth why the sda market is becoming foundational to space operations (spaceinsider.tech)](https://spaceinsider.tech/2025/08/22/ground-truth-why-the-sda-market-is-becoming-foundational-to-space-operations/)
- [Leolabs lands interagency contract to feed tracss and track adversarial spacecraft (spacenews.com)](https://spacenews.com/leolabs-lands-interagency-contract-to-feed-tracss-and-track-adversarial-spacecraft/)
- [Osc jco license leolabs object catalog (leolabs.space)](https://leolabs.space/press/osc-jco-license-leolabs-object-catalog/)
- [Slingshot aerospace wins 69 2 million u s space force contract to advance ai powered mission readiness for space defense (slingshot.space)](https://www.slingshot.space/news/slingshot-aerospace-wins-69-2-million-u-s-space-force-contract-to-advance-ai-powered-mission-readiness-for-space-defense)
- [Space force darc radar site wales cawdor barracks northrop grumman (defensescoop.com)](https://defensescoop.com/2024/08/23/space-force-darc-radar-site-wales-cawdor-barracks-northrop-grumman/)
- [Space force reorganizing to add commercial space monitoring data analysis into operations (breakingdefense.com)](https://breakingdefense.com/2025/09/space-force-reorganizing-to-add-commercial-space-monitoring-data-analysis-into-operations/)
- [Space force spending could hit 40b in (airandspaceforces.com)](https://www.airandspaceforces.com/space-force-spending-could-hit-40b-in-2026/)
- [Space situational awareness market report (grandviewresearch.com)](https://www.grandviewresearch.com/industry-analysis/space-situational-awareness-market-report)
- [Space situational awareness market to reach usd 2 79 billion by 2030 growing at 10 0 cagr says marketsandmarkets (globenewswire.com)](https://www.globenewswire.com/news-release/2026/08/25/3350421/0/en/space-situational-awareness-market-to-reach-usd-2-79-billion-by-2030-growing-at-10-0-cagr-says-marketsandmarkets.html)
- [Space threat assessment (csis.org)](https://www.csis.org/analysis/space-threat-assessment-2025)
- [Space traffic management market (astuteanalytica.com)](https://www.astuteanalytica.com/industry-report/space-traffic-management-market)
- [Spacex lowering orbits of 4 400 starlink satellites for safetys sake (space.com)](https://www.space.com/space-exploration/satellites/spacex-lowering-orbits-of-4-400-starlink-satellites-for-safetys-sake)
- [U s gssap satellites execute geo handoff to monitor chinas shijian 29 spacecraft (spacenews.com)](https://spacenews.com/u-s-gssap-satellites-execute-geo-handoff-to-monitor-chinas-shijian-29-spacecraft/)

### Government programs and customers

- [AMOS 2025 technical paper, Mashiku](https://amostech.com/TechnicalPapers/2025/ConjunctionRPO/Mashiku.pdf)
- [Bluestaq wins 280 million space force contract to expand space data catalog (spacenews.com)](https://spacenews.com/bluestaq-wins-280-million-space-force-contract-to-expand-space-data-catalog/)
- [Dynamic space operations (airandspaceforces.com)](https://www.airandspaceforces.com/article/dynamic-space-operations/)
- [esa.int](https://www.esa.int/Space_Safety/Boost_in_funding_expands_Space_Safety_programme)
- [Eu sst continues growing and preparing future (eusst.eu)](https://www.eusst.eu/newsroom/news/eu-sst-continues-growing-and-preparing-future)
- [Future space domain awareness needs national security space (csis.org)](https://www.csis.org/analysis/future-space-domain-awareness-needs-national-security-space)
- [Golden dome budget plan increase space capabilities guetlein (defensescoop.com)](https://defensescoop.com/2026/03/17/golden-dome-budget-plan-increase-space-capabilities-guetlein/)
- [Pentagon report space force atlas program falls short of decommissioning targets (satnews.com)](https://satnews.com/2026/03/21/pentagon-report-space-force-atlas-program-falls-short-of-decommissioning-targets/)
- [Space development agency makes awards to build 72 tracking layer satellites for tranche 3 (sda.mil)](https://www.sda.mil/space-development-agency-makes-awards-to-build-72-tracking-layer-satellites-for-tranche-3/)
- [Space force chief current satellite tracking too slow for modern threats (spacenews.com)](https://spacenews.com/space-force-chief-current-satellite-tracking-too-slow-for-modern-threats/)
- [Space force declares atlas space domain awareness software operational (breakingdefense.com)](https://breakingdefense.com/2025/09/space-force-declares-atlas-space-domain-awareness-software-operational/)
- [Space force unveils multi front push to fix its unified data library (breakingdefense.com)](https://breakingdefense.com/2025/03/space-force-unveils-multi-front-push-to-fix-its-unified-data-library/)
- [Spacewerx funds bosonq psi federal quantum inspired ai for space domain awareness (militaryaerospace.com)](https://www.militaryaerospace.com/sensors/article/55391793/spacewerx-funds-bosonq-psi-federal-quantum-inspired-ai-for-space-domain-awareness)
- [To infinity and beyond new space force unit to monitor xgeo beyond earths orbit (breakingdefense.com)](https://breakingdefense.com/2022/04/to-infinity-and-beyond-new-space-force-unit-to-monitor-xgeo-beyond-earths-orbit/)
- [Where space force budget commercial services (csis.org)](https://www.csis.org/analysis/where-space-force-budget-commercial-services)

### Commercial SSA/SDA competitors

- [Anduril reaches agreement to acquire exoanalytic solutions to accelerate space domain awareness and missile defense capabilities (anduril.com)](https://www.anduril.com/news/anduril-reaches-agreement-to-acquire-exoanalytic-solutions-to-accelerate-space-domain-awareness-and-missile-defense-capabilities)
- [Arbitration panel rules for spire in dispute with northstar earth northstar says its still owed 4 replacement satellites (spaceintelreport.com)](https://www.spaceintelreport.com/arbitration-panel-rules-for-spire-in-dispute-with-northstar-earth-northstar-says-its-still-owed-4-replacement-satellites/)
- [Managing space domain awareness data has become a greater challenge than collecting it (spacenews.com)](https://spacenews.com/managing-space-domain-awareness-data-has-become-a-greater-challenge-than-collecting-it/)
- [Neuraspace secures 15.6 million to scale ai driven space traffic management and defence grade space domain awareness (blog.neuraspace.com)](https://blog.neuraspace.com/neuraspace-secures-15.6-million-to-scale-ai-driven-space-traffic-management-and-defence-grade-space-domain-awareness)
- [Vyoma announces the successful commissioning of the flamingo 1 platform (vyoma.space)](https://vyoma.space/news-items/vyoma-announces-the-successful-commissioning-of-the-flamingo-1-platform)

### Procurement pathways and business models

- [Afwerx sbir guide (getcada.com)](https://www.getcada.com/insights/afwerx-sbir-guide)
- [afwerx.com](https://afwerx.com/divisions/sbir-sttr/phase-iii/)
- [AMOS 2024 technical paper, Golf](https://amostech.com/TechnicalPapers/2024/Featured/Golf.pdf)
- [Calvelli details plans to better integrate unified data library into space force ops (breakingdefense.com)](https://breakingdefense.com/2024/06/calvelli-details-plans-to-better-integrate-unified-data-library-into-space-force-ops/)
- [diu.mil](https://diu.mil/work-with-us)
- [Gao 25 106856.pdf (gao.gov)](https://www.gao.gov/assets/gao-25-106856.pdf)
- [incubed.esa.int](https://incubed.esa.int/how-to-apply/)
- [New afwerx sbir opportunity 25 5 e release 10 open topic phase i now open (ebhoward.com)](https://www.ebhoward.com/new-afwerx-sbir-opportunity-25-5-e-release-10-open-topic-phase-i-now-open/)
- [New export control rules present key regulatory changes for space (hklaw.com)](https://www.hklaw.com/en/insights/publications/2024/11/new-export-control-rules-present-key-regulatory-changes-for-space)
- [Noaa proposes terminating tracss program (payloadspace.com)](https://payloadspace.com/noaa-proposes-terminating-tracss-program/)
- [Release spaceflux awarded multimillion pound uk government contracts to deliver sovereign space surveillance and tracking (spaceflux.io)](https://spaceflux.io/press-release-spaceflux-awarded-multimillion-pound-uk-government-contracts-to-deliver-sovereign-space-surveillance-and-tracking/)
- [Spacewerx announces stratfi awardees (payloadspace.com)](https://payloadspace.com/spacewerx-announces-2026-stratfi-awardees/)
- [Spacewerx selects eight companies for 440 million in public private partnerships (spacenews.com)](https://spacenews.com/spacewerx-selects-eight-companies-for-440-million-in-public-private-partnerships/)
- [spacewerx.us](https://spacewerx.us/accelerate/stratfi-tacfi/)
- [ssc.spaceforce.mil](https://www.ssc.spaceforce.mil/Portals/3/SDA%20Briefings/06.%20UDL%20Overview%20Overview.pdf)
- [True anomaly lands 17 million us space force contract for space domain awareness (prnewswire.com)](https://www.prnewswire.com/news-releases/true-anomaly-lands-17-million-us-space-force-contract-for-space-domain-awareness-301934799.html)
- [What is the spec ota (nstxl.org)](https://nstxl.org/faq-items/what-is-the-spec-ota/)

### Debris, constellations and macro trends

- [Global counterspace capabilities report (swfound.org)](https://www.swfound.org/publications-and-reports/2026-global-counterspace-capabilities-report)
- [Intelsat 33e breakup (keeptrack.space)](https://keeptrack.space/deep-dive/intelsat-33e-breakup)
- [Kessler syndrome space debris (spectrum.ieee.org)](https://spectrum.ieee.org/kessler-syndrome-space-debris)
- [Orbital launch attempts by country (payloadspace.com)](https://payloadspace.com/2025-orbital-launch-attempts-by-country/)
- [planetary.org](https://www.planetary.org/space-missions/clps)
- [Spacex china drive new record for orbital launches in (spacenews.com)](https://spacenews.com/spacex-china-drive-new-record-for-orbital-launches-in-2025/)
- [The space report q2 (spacefoundation.org)](https://www.spacefoundation.org/2025/07/22/the-space-report-2025-q2/)
- [What are biggest space threats (csis.org)](https://www.csis.org/analysis/what-are-biggest-space-threats-2026)

### BQP company and product

- [2023 03 22 BosonQ Psi Joins IBM Quantum Network (in.newsroom.ibm.com)](https://in.newsroom.ibm.com/2023-03-22-BosonQ-Psi-Joins-IBM-Quantum-Network/)
- [Bosonq psi raises 3m in seed funding to advance quantum inspired simulation technology (thequantuminsider.com)](https://thequantuminsider.com/2024/12/17/bosonq-psi-raises-3m-in-seed-funding-to-advance-quantum-inspired-simulation-technology/)
- [Bqp awarded first federal contract to advance quantum assisted ai for space domain awareness (prnewswire.com)](https://www.prnewswire.com/news-releases/bqp-awarded-first-federal-contract-to-advance-quantum-assisted-ai-for-space-domain-awareness-302828047.html)
- [Bqp ceo abhishek chopra honored (finance.yahoo.com)](https://finance.yahoo.com/news/bqp-ceo-abhishek-chopra-honored-133900695.html)
- [Bqp raises 5m oversubscribed seed round following pilot agreement with air force research lab for quantum accelerated digital twin platform (prnewswire.com)](https://www.prnewswire.com/news-releases/bqp-raises-5m-oversubscribed-seed-round-following-pilot-agreement-with-air-force-research-lab-for-quantum-accelerated-digital-twin-platform-302507885.html)
- [Bqp secures spacewerx funding for quantum assisted ai for space domain awareness (satellitetoday.com)](https://www.satellitetoday.com/technology/2026/07/17/bqp-secures-spacewerx-funding-for-quantum-assisted-ai-for-space-domain-awareness/)
- [bqpsim.com](https://www.bqpsim.com/about-us)
- [computer.org](https://www.computer.org/csdl/proceedings-article/qce/2024/413701b707/23oq1ElF4zK)

### Other material consulted

- [AFRL Oracle program](https://afresearchlab.com/cislunar-highway-patrol-system-chps/)
- [AMOS  technical paper, amostech.com](https://amostech.com/)
- [AMOS 2024 technical paper, Allen](https://amostech.com/TechnicalPapers/2024/Poster/Allen.pdf)
- [AMOS 2025 technical paper, Latif](https://amostech.com/TechnicalPapers/2025/Poster/Latif.pdf)
- [AMOS 2026 technical paper, AMOS-2026-Program](https://amostech.com/wp-content/uploads/2026/08/AMOS-2026-Program.pdf)
- [Beyond potpourri of sensors saltzman pushes holistic approach to space domain awareness (breakingdefense.com)](https://breakingdefense.com/2025/09/beyond-potpourri-of-sensors-saltzman-pushes-holistic-approach-to-space-domain-awareness/)
- [BMC3I TAP Lab](https://bmc3itaplab.space/)
- [business.defense.gov](https://business.defense.gov/cmmc)
- [CCSDS publications](https://public.ccsds.org/Publications/default.aspx)
- [dote.osd.mil](https://www.dote.osd.mil/Portals/97/pub/reports/FY2025/af/2025space-c2.pdf)
- [Eradrive raises 5 3 million for software hardware kits to enhance satellite autonomy (spacenews.com)](https://spacenews.com/eradrive-raises-5-3-million-for-software-hardware-kits-to-enhance-satellite-autonomy/)
- [find-tender.service.gov.uk](https://find-tender.service.gov.uk/Notice/003789-2026)
- [gao.gov](https://www.gao.gov/products/gao-26-107085)
- [Gstp e1 compendia 2028.pdf (wiin-io.s3.eu-west-3.amazonaws.com)](https://wiin-io.s3.eu-west-3.amazonaws.com/620bc84520454e78d575df56/files/231d72fdac/gstp-e1-compendia-2026-2028.pdf)
- [House bill restores funding for tracss (spacenews.com)](https://spacenews.com/house-bill-restores-funding-for-tracss/)
- [If funding falters theres no golden dome general warns (breakingdefense.com)](https://breakingdefense.com/2026/08/if-funding-falters-theres-no-golden-dome-general-warns/)
- [inspace.gov.in](https://www.inspace.gov.in/inspace/sys_attachment.do?sys_id=bb786ff33b6d0b1063ff0a34c3e45ab5)
- [Kronos family of systems commercial solutions open fa8806 26 s 0001 o 0e178 (highergov.com)](https://www.highergov.com/contract-opportunity/kronos-family-of-systems-commercial-solutions-open-fa8806-26-s-0001-o-0e178/)
- [Optimize space defense missions bqphy (bqpsim.com)](https://www.bqpsim.com/optimize-space-defense-missions-bqphy)
- [patents.google.com](https://patents.google.com/patent/US20260105218A1/en)
- [Pentagon announces immediate suspension of cmmc mandates (breakingdefense.com)](https://breakingdefense.com/2026/07/pentagon-announces-immediate-suspension-of-cmmc-mandates/)
- [Sda director says constellations tranche 1 checkouts progressing faster after a slow start (satellitetoday.com)](https://www.satellitetoday.com/government-military/2026/08/24/sda-director-says-constellations-tranche-1-checkouts-progressing-faster-after-a-slow-start/)
- [SDA TAP LAB CATALYST CAMPUS MINI ACCELERATOR WELCOMES SECOND COHORT OF INNOVATIVE COMPANIES TO COLORADO SPRINGS (globenewswire.com)](https://www.globenewswire.com/news-release/2025/07/14/3115259/0/en/SDA-TAP-LAB-CATALYST-CAMPUS-MINI-ACCELERATOR-WELCOMES-SECOND-COHORT-OF-INNOVATIVE-COMPANIES-TO-COLORADO-SPRINGS.html)
- [Sda tap lab evolves into bmc3i tap lab and in partnership with catalyst campus launches a new multi phased program to accelerate mission focused technology development (spacenews.com)](https://spacenews.com/sda-tap-lab-evolves-into-bmc3i-tap-lab-and-in-partnership-with-catalyst-campus-launches-a-new-multi-phased-program-to-accelerate-mission-focused-technology-development/)
- [sda.mil](https://www.sda.mil/custody/)
- [Shield ai and sedaro demonstrate trusted autonomy capabilities on novi satellite (shield.ai)](https://shield.ai/shield-ai-and-sedaro-demonstrate-trusted-autonomy-capabilities-on-novi-satellite/)
- [Space commerce official tracss continues despite budget uncertainty (fedscoop.com)](https://fedscoop.com/space-commerce-official-tracss-continues-despite-budget-uncertainty/)
- [Space force battle management lab operators (airandspaceforces.com)](https://www.airandspaceforces.com/space-force-battle-management-lab-operators/)
- [Space force testing ai automate ops (airandspaceforces.com)](https://www.airandspaceforces.com/space-force-testing-ai-automate-ops/)
- [spaceforce.mil](https://www.spaceforce.mil/Portals/2/Documents/SAF_2025/Space_Warfighting_-_A_Framework_for_Planners_BLK2_%28final_20250410%29.pdf)
- [spaceforce.mil](https://www.spaceforce.mil/Portals/2/Documents/White_Paper_Summary_of_Competitive_Endurance.pdf)
- [Spacex offers space safety service for satellite operators (spacenews.com)](https://spacenews.com/spacex-offers-space-safety-service-for-satellite-operators/)

---

# Verification record

The deck's own claim ledger (`research-audit.md` in the presentation working
directory) records the checks behind these citations. In summary:

- Date-sensitive claims were checked on 2026-08-28 and again during the
  2026-08-31 closeout.
- A final link check covered 26 unique deck URLs. Twenty-five returned HTTP
  200. The Space Systems Command UDL article returned HTTP 403 to the automated
  client, which is consistent with site access control rather than a dead page.
- The two ESA PDFs were downloaded and passed a `qpdf --check` integrity test,
  as did the AMOS 2023 paper.
- Live re-reads on 31 August confirmed the ESA tracked-object count and update
  date, the Gannon storm qualification, the Kayhan machine-to-machine claim,
  the two conformal-prediction guarantees, and the wording on the BQP product
  page.

## Corrections found while compiling this list

Publisher records were re-checked on 2026-09-03 while writing these entries.
Two citations in the deck's own notes need correcting, and the corrected form
is what appears above.

- The Gannon storm paper is Journal of Spacecraft and Rockets **61(5)**, pages
  1412 to 1416. The deck's claim ledger records it as 61(6). The volume, year
  and content are unaffected.
- The control-distance-metric paper has **three** authors: Holzinger, Scheeres
  and Alfriend, Journal of Guidance, Control, and Dynamics 35(4), 1312 to 1325,
  2012. The deck cites it as "Holzinger and Scheeres".

Two further cautions on the primary records themselves. The AETHER-P3 journal
entry lists an article number of `e2026SW000000`, which is an author
placeholder rather than an assigned number, so cite the DOI instead. The arXiv
record for the stochastic-hybrid maneuver paper carries no journal reference,
so no journal citation should be attached to it.

Two things are worth carrying forward when reusing this list. First, several
sources are explicitly date-sensitive: the ESA statistics page, the TraCSS
programme pages, the funding and cohort routes, and every company product page.
Re-check them before quoting a number. Second, a source being primary does not
make the deck's use of it a measured BQP result. Entries marked as study
results, preprints, or company-reported stay that way.

# Referenced without a link

**DoD Directive 3000.09, Autonomy in Weapon Systems.** named on the authority  
backup slide only to state what it does *not* do: it is not a blanket rule for
non-weapon collision avoidance.

# Where these came from

- `presentation.tex`, the deck source, parsed for every `\href` target: 109
  citation occurrences across 47 unique URLs, of which 40 also appear on the
  deck's own "Sources and evidence labels" slide.
- `research-audit.md`, the claim ledger, for attributions, qualifications, and
  the verification record.
- `research-findings.md` and `research-technical-depth.md` for the background
  corpus in Part 2.
