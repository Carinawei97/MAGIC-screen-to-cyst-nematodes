Scripts for the manuscript: [3D printing and deep learning enable holistic and dynamic analyses of tens of thousands of parasites infecting hundreds of genotypes]

Welcome to the repository for the scripts used in our manuscript, "3D printing and deep learning enable holistic and dynamic analyses of tens of thousands of parasites infecting hundreds of genotypes”. This repository contains code written in Python and R for various tasks, including controlling an imaging tower, working with QR codes, deep-learning model Fast-nema for nematode capture ,statistical modeling using the BLUES model, and data visualisation, as well as STL files for 3D printing. This workshop was developmed by Siyuan Wei, Jie Zhou, Olaf Kranse, and Sebastian, Eves-van den Akker.

Table of Contents

	1.	Project Description
	2.	Scripts Overview
	3.	Dependencies
	4.	How to Use
	5.	Contact
	6.	Acknowledgments
	7.	License

Project Description

This repository includes the scripts developed for my our manuscript, focusing on the 3D printed and Deep-learning powered high-throughput phenotyping platform to automatic capture of the nematode-centric phenotypes. The scripts were used for:

	•	Controlling a custom-built imaging tower.
	•	Reading, storing, and generating QR codes.
 	•	The deep-learning model Fast-nema for recognising nematode phenotypes: including number, size, shape, colour.
	•	Implementing the BLUES model for pairwise comparisons.
	•	Creating publication-quality visualisations.

These scripts were developed in collaboration with my supervisor and colleagues.

Scripts Overview

Python Scripts

	1.	Imaging Tower Control:
	•	Scripts for controlling the hardware and capturing images.
	2.	QR Code Handling:
	•	Generating QR codes for labeling samples.
	•	Reading and storing information from QR codes.

R Scripts

	1.	Statistical Modeling:
	•	Implementation of the BLUES model for pairwise comparisons.
	2.	Visualisation:
	•	Scripts for plotting data and creating publication-ready figures.

STL files: for 3D printing the automatic imaging machine.

Dependencies

Make sure you have the following software and libraries installed:

Python

	•	numpy
	•	opencv-python
	•	qrcode
	•	pandas
	•	Other dependencies (list specific libraries used in your scripts).

R

	•	ggplot2
	•	lme4 (if used for BLUES model)
	•	Other required packages.

How to Use

	1.	Clone the repository:
 git clone https://github.com/Carinawei97/MAGIC-screen-to-cyst-nematodes.git
 cd MAGIC-screen-to-cyst-nematodes

 	2.	Explore the Python and R directories to access the scripts:
	•	Python scripts for imaging tower control and QR code handling.
	•	R scripts for modeling and plotting.
	3.	Follow comments in each script for usage details. If you have any questions, feel free to reach out!

Contact

If you have any questions or suggestions regarding the code, please open an issue or contact me via GitHub. I’m happy to collaborate or assist.

Acknowledgements

This work was developed as part of the manuscript under the guidance of Sebastian Eves-van de Akker and with contributions from Olaf kranse and Jie Zhou. Thank you for your support and feedback throughout this project.

License

This project is licensed under the GNU General Public License v3.0.

You are free to use, modify, and distribute this code under the terms of the GPL v3.0 license. For more details, please refer to the LICENSE file included in this repository or visit the GNU GPL v3.0 for the full text of the license.

Adding the LICENSE File

To make your repository fully compliant with the GNU GPL v3.0, create a file named LICENSE in your repository root and add the full text of the GPL v3.0. You can copy the text from this link.
 
