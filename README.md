# nest-soc
The **nest-soc** package aims to help engineers and scientists to have access to state-of-art models for solid oxide cells performance and degradation.

## Preview version
The **nest-soc** package is a work in progress and this is a early preview version showcasing some use examples:
1. Polarization curve simulation for H2O-H2 operation;
2. Polarization curve simulation for syngas mixture operation;
3. Voltage degradation prediction for solid oxide fuel cell operation
4. Polarization curve simulation for CO2-CO operation and carbon formation prdiction
5. Electrochemical impedance spectrum (EIS) simulation

### Running preview
1. Install dependencies specified in the [requirements file](requirements.txt) by running the following command (from the nest-soc root folder):

```bash
pip install -r requirements.txt
```

2. Run main.py file (from the nest-soc root folder):
```bash
python main.py
```
## How to cite
For the **package**, you can use the following citation:
```bibtex
@misc{NogueiraNakashima2026,
   author = {R. Nogueira Nakashima and J. Beyrami},
   doi = {10.11583/DTU.32012556},
   publisher = {Technical University of Denmark},
   title = {nest-soc: Next-gen electrochemical tools - solid oxide cells},
   url = {https://doi.org/10.11583/DTU.32012556},
   year = {2026}
}
```

For the modeling methods you can cite the following:
```bibtex
@article{Beyrami2024,
   author = {Javid Beyrami and Rafael Nogueira Nakashima and Arash Nemati and Henrik Lund Frandsen},
   doi = {10.1016/j.ecmx.2024.100653},
   issn = {25901745},
   journal = {Energy Conversion and Management: X},
   month = {7},
   pages = {100653},
   title = {Degradation modeling in solid oxide electrolysis systems: A comparative analysis of operation modes},
   volume = {23},
   year = {2024}
}
```


## Contact info
For collaborations, inquires, or requests, feel free to contact:
* Rafael Nogueira Nakashima ([rafnn@dtu.dk](mailto:rafnn@dtu.dk))
* Javid Beyrami ([jabey@dtu.dk](mailto:jabey@dtu.dk))

## Development partners
* HD Hyundai ERC GmbH: https://hdhyundai-erc.com/

## Acknowledgement
This work has been funded by [HD Hyundai ERC GmbH](https://hdhyundai-erc.com/) through the NEST project.

