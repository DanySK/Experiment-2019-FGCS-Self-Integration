# Partitioned Integration and Coordination via Self-organising Coordination Regions

## Experimental results of the second case study

This repository contains code and instruction to reproduce the experiments presented in the paper *Partitioned Integration and Coordination via Self-organising Coordination Regions* by Danilo Pianini, Roberto Casadei, and Mirko Viroli; submitted to Elsevier's [Future Generation Computer Systems](https://www.journals.elsevier.com/future-generation-computer-systems) journal, in the [Special Issue on Self-integrating Systems: Mastering Continuous Change](https://www.journals.elsevier.com/future-generation-computer-systems/call-for-papers/self-integrating-systems-mastering-continuous-change).

## Requirements

In order to run the experiments, the Java Development Kit 11 is required.
We test using OpenJ9 11 and 13, and OpenJDK 11 and 13.
Original testing was performed with OpenJDK 11 and OpenJ9 11.
It is known that this project does not work (due to a compatibility issue of the version of Protelis in use) to any JDK from 14 on.

In order to produce the charts, Python 3 is required.
We recommend Python 3.6.1,
but it is very likely that any Python 3 version,
and in particular any later version will serve the purpose just as well.
The recommended way to obtain it is via [pyenv](https://github.com/pyenv/pyenv).

The experiments have been designed and tested under Linux.
However, we have some muliplatform build automation in place.
Everything should run on any recent Linux, MacOS X, and Windows setup.

### Reference machine

We provide a reference Travis CI configuration to maintain reproducibility over time.
While this image: [![Build Status](https://travis-ci.org/DanySK/Experiment-2019-FGCS-Self-Integration.svg?branch=master)](https://travis-ci.org/DanySK/Experiment-2019-FGCS-Self-Integration)
is green, the experiment is being maintained and,
by copying the configuration steps we perform for Travis CI in the `.travis.yml` file,
you should be able to re-run the experiment entirely.

### Automatic releases

Charts are remotely generated and made available on the project release page.
[The latest release](https://github.com/DanySK/DanySK/Experiment-2019-SCP-Graph-Statistics/releases/latest)
allows for quick retrieval of the latest version of the charts.

## Running the simulations

A graphical execution of the simulation can be started by issuing the following command
`./gradlew showAll`.
Simulation on the graphical interface can be started with <kbd>P</kbd>,
for further details please refer to the Alchemist Simulator guide.
Windows users may try using the `gradlew.bat` script as a replacement for `gradlew`.

The whole simulation batch can be executed by issuing `./gradlew runAll`.
**Be aware that it may take a very long time**, from several hours to weeks, depending on your hardware.
If you are under Linux, the system tries to detect the available memory and CPUs automatically, and parallelize the work.

## Generating the charts

In order to speed up the process for those interested in observing and manipulating the existing data,
we provide simulation-generated data directly in the repository.
Generating the charts is matter of executing the `process.py` script.
The enviroment is designed to be used in conjunction with pyenv.

### Python environment configuration

The following guide will start from the assumption that pyenv is installed on your system.
First, install Python by issuing

``pyenv install --skip-existing 3.6.1``

Now, configure the project to be interpreted with exactly that version:

``pyenv local 3.6.1``

Update the `pip` package manager and install the required dependencies.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Data processing and chart creation

This section assumes you correctly completed the required configuration described in the previous section.
In order for the script to execute, you only need to launch the actual process by issuing `python process.py`
