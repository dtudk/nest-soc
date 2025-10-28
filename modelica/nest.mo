package nest
  package HydrogenWater "Mixture of H2 and H2O"
    extends Modelica.Media.IdealGases.Common.MixtureGasNasa(mediumName = "HydrogenWater", data = {Modelica.Media.IdealGases.Common.SingleGasesData.H2, Modelica.Media.IdealGases.Common.SingleGasesData.H2O}, fluidConstants = {Modelica.Media.IdealGases.Common.FluidData.H2, Modelica.Media.IdealGases.Common.FluidData.H2O}, substanceNames = {"Hydrogen", "Water"}, reference_X = {0.5, 0.5});
    annotation(
      Documentation(info = "<html>
  
  </html>"),
      uses(Modelica(version = "4.0.0")));
  end HydrogenWater;

  model ocv "Thermodynamic voltage for H2/H2O reaction based on Nernst equation for ideal gases"
    // Constants
    constant Real n_e = 2 "electrons transfer per reaction";
    // mol/mol
    constant Modelica.Units.SI.Pressure P_0 = 1E5 "Reference pressure";
    // Pa
    // Parameter
    parameter Modelica.Units.SI.Area A "Segment area";
    // m2
    // Ports
    Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_a h annotation(
      Placement(transformation(origin = {100, 100}, extent = {{-110, -10}, {-90, 10}}), iconTransformation(origin = {100, 100}, extent = {{-110, -10}, {-90, 10}})));
    Modelica.Blocks.Interfaces.RealInput P_H2(unit = "Pa") annotation(
      Placement(transformation(origin = {-60, -120}, extent = {{-20, -20}, {20, 20}}, rotation = 90), iconTransformation(origin = {-60, -120}, extent = {{-20, -20}, {20, 20}}, rotation = 90)));
    Modelica.Blocks.Interfaces.RealInput P_H2O(unit = "Pa") annotation(
      Placement(transformation(origin = {0, -120}, extent = {{-20, -20}, {20, 20}}, rotation = 90), iconTransformation(origin = {0, -120}, extent = {{-20, -20}, {20, 20}}, rotation = 90)));
    Modelica.Blocks.Interfaces.RealInput P_O2(unit = "Pa") annotation(
      Placement(transformation(origin = {60, -120}, extent = {{-20, -20}, {20, 20}}, rotation = 90), iconTransformation(origin = {60, -120}, extent = {{-20, -20}, {20, 20}}, rotation = 90)));
    // Extensions
    extends Modelica.Electrical.Analog.Interfaces.OnePort;
  equation
// Nernst
    -v = (1.2566 - 2.862E-8*h.T^2 - 2.296E-4*h.T) + Modelica.Constants.R*h.T/(n_e*Modelica.Constants.F)*log(P_H2/P_H2O*(P_O2/P_0)^0.5);
// Reaction heat
    h.Q_flow = -(1.2346 - 1.659E-8*h.T^2 + 6.633E-5*h.T)*p.i;
// OBS: Quadratic approximation of thermodynamic properties for H2/H2O reaction based on NASA properties (25 C < T < 1000 C, error < 0.1%)
    annotation(
      uses(Modelica(version = "4.0.0")),
      Documentation(__OpenModelica_infoHeader = "<html><head></head><body><h1>nest.ocv</h1></body></html>"),
      Icon(graphics = {Line(origin = {0, 1}, points = {{-100, -1}, {100, -1}}, color = {0, 0, 255}), Ellipse(lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.CrossDiag, extent = {{-47, 47}, {47, -47}})}));
  end ocv;

  model ohmic "Overpotential associated with charge transfer (ohmic) for a cell layer"
    parameter Modelica.Units.SI.Length delta "Layer thickness";
    // m2
    parameter Modelica.Units.SI.Conductivity sigma_0 "Reference conductivity";
    // S/m
    parameter Real theta "Temperature order";
    // -
    parameter Modelica.Units.SI.MolarEnthalpy E_act "Activation energy";
    // J/mol
    parameter Modelica.Units.SI.Area A "Segment area";
    // m2
    Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_a h annotation(
      Placement(transformation(origin = {100, 100}, extent = {{-110, -10}, {-90, 10}}), iconTransformation(origin = {100, 100}, extent = {{-110, -10}, {-90, 10}})));
    extends Modelica.Electrical.Analog.Interfaces.OnePort;
  equation
// Ohm's law + General conductivity model
    v = (delta/(sigma_0*h.T^theta*exp(-E_act/(Modelica.Constants.R*h.T))))*p.i/A;
// Joule heat
    h.Q_flow = -v*p.i;
    annotation(
      uses(Modelica(version = "4.0.0")),
      Icon(graphics = {Line(origin = {0, 1}, points = {{-100, -1}, {100, -1}}, color = {0, 0, 255}), Rectangle(lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.Backward, extent = {{-60, 20}, {60, -20}})}));
  end ohmic;

  model activation "Overpotential associated with reaction kinetics (activation) for an electrode"
    // Constants
    constant Real n_e = 2 "electrons transfer per reaction";
    // mol/mol
    constant Modelica.Units.SI.Pressure P_0 = 1E5;
    // Inputs
    parameter Modelica.Units.SI.CurrentDensity j_0_inf "Reference exchange current density";
    // A/m2
    parameter Real theta "Temperature order";
    // -
    parameter Real a "P_H2 order";
    // -
    parameter Real b "P_H2O order";
    // -
    parameter Real m "P_O2 order";
    // -
    parameter Modelica.Units.SI.MolarEnthalpy E_act "Activation energy";
    // J/mol
    parameter Real alpha "Charge transfer coefficient";
    parameter Modelica.Units.SI.Area A "Segment area";
    // m2
    // Ports
    Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_a h annotation(
      Placement(transformation(origin = {100, 100}, extent = {{-110, -10}, {-90, 10}}), iconTransformation(origin = {100, 100}, extent = {{-110, -10}, {-90, 10}})));
    Modelica.Blocks.Interfaces.RealInput P_H2(unit = "Pa") annotation(
      Placement(transformation(origin = {-60, -120}, extent = {{-20, -20}, {20, 20}}, rotation = 90), iconTransformation(origin = {-60, -120}, extent = {{-20, -20}, {20, 20}}, rotation = 90)));
    Modelica.Blocks.Interfaces.RealInput P_H2O(unit = "Pa") annotation(
      Placement(transformation(origin = {0, -120}, extent = {{-20, -20}, {20, 20}}, rotation = 90), iconTransformation(origin = {0, -120}, extent = {{-20, -20}, {20, 20}}, rotation = 90)));
    Modelica.Blocks.Interfaces.RealInput P_O2(unit = "Pa") annotation(
      Placement(transformation(origin = {60, -120}, extent = {{-20, -20}, {20, 20}}, rotation = 90), iconTransformation(origin = {60, -120}, extent = {{-20, -20}, {20, 20}}, rotation = 90)));
    // Extension
    extends Modelica.Electrical.Analog.Interfaces.OnePort;
  equation
// Butler-Volmer + Generic exchange current density
    p.i = A*j_0_inf*h.T^theta*(P_H2/P_0)^a*(P_H2O/P_0)^b*(P_O2/P_0)^m*exp(-E_act/(Modelica.Constants.R*h.T))*(exp(alpha*n_e*Modelica.Constants.F*v/(Modelica.Constants.R*h.T)) - exp(-(1 - alpha)*n_e*Modelica.Constants.F*v/(Modelica.Constants.R*h.T)));
// Joule heat
    h.Q_flow = -v*p.i;
    annotation(
      uses(Modelica(version = "4.0.0")),
      Diagram,
      Icon(graphics = {Line(origin = {0, 1}, points = {{-100, -1}, {100, -1}}, color = {0, 0, 255}), Rectangle(lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.Forward, extent = {{-20, 60}, {20, -60}})}));
  end activation;

  model porousDiffusion "Mass transfer limitations associated with diffusion in porous electrodes/support layers"
    // Constants
    type MolarMass_gmol = Real(unit = "g/mol", min = 0);
    type DiffusionVolume = Real(unit = "cm3", min = 0);
    constant Real n_e = 2;
    // mol/mol
    // Parameters
    parameter Real v_A = 1 "Stoichiometric coeff. - A";
    parameter Real v_B = -1 "Stoichiometric coeff. - B";
    parameter Modelica.Units.SI.Area A = 1 "Segment area";
    parameter Modelica.Units.SI.Length delta = 310e-6 "Layer thickness";
    parameter Modelica.Units.SI.Length delta_ch = 0.01 "Channel height";
    parameter Real epsilon = 0.4 "Porosity";
    parameter Real tau = 3 "Tortuosity";
    parameter Modelica.Units.SI.Length r_p = 3e-7 "Porous radius";
    parameter MolarMass_gmol M_A = 2.016 "Molar mass - A";
    parameter MolarMass_gmol M_B = 18.02 "Molar mass - B";
    parameter DiffusionVolume V_A = 6.12 "Fuller volume - A";
    parameter DiffusionVolume V_B = 13.1 "Fuller volume - B";
    parameter Integer n = 10 "Number of segments";
    parameter Modelica.Units.SI.CoefficientOfHeatTransfer k = 1 "Heat transfer coeff.";
    // Initial conditions
    parameter Modelica.Units.SI.Pressure P_A_initial = 0.9E5 "Initial pressure - A";
    parameter Modelica.Units.SI.Pressure P_B_initial = 0.1E5 "Initial pressure - B";
    // Calculated variables
    Modelica.Units.SI.Pressure P_A[n];
    Modelica.Units.SI.Pressure P_B[n];
    Modelica.Units.SI.Pressure P_A_bulk;
    Modelica.Units.SI.Pressure P_B_bulk;
    Modelica.Units.SI.Temperature T_gas;
    // Ports
    replaceable package Medium = Modelica.Media.Interfaces.PartialMedium "Medium in the component" annotation(
      choicesAllMatching = true);
    Modelica.Fluid.Interfaces.FluidPort_a port_a(redeclare package Medium = Medium) annotation(
      Placement(transformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-100, 0}, extent = {{-10, -10}, {10, 10}})));
    Modelica.Fluid.Interfaces.FluidPort_b port_b(redeclare package Medium = Medium) annotation(
      Placement(transformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {100, 0}, extent = {{-10, -10}, {10, 10}})));
    Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_a port_h annotation(
      Placement(transformation(origin = {0, 98}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-50, 100}, extent = {{-10, -10}, {10, 10}})));
    Modelica.Blocks.Interfaces.RealInput current(unit = "A") annotation(
      Placement(transformation(origin = {0, -120}, extent = {{-20, -20}, {20, 20}}, rotation = 90), iconTransformation(origin = {48, 120}, extent = {{-20, -20}, {20, 20}}, rotation = -90)));
    Modelica.Blocks.Interfaces.RealOutput P_A_tpb(unit = "Pa") annotation(
      Placement(transformation(origin = {-56, -134}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-50, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    Modelica.Blocks.Interfaces.RealOutput P_B_tpb(unit = "Pa") annotation(
      Placement(transformation(origin = {50, -102}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {50, -110}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
  protected
    Modelica.Units.SI.DiffusionCoefficient D_eff_A;
    Modelica.Units.SI.DiffusionCoefficient D_eff_B;
    Modelica.Units.SI.DiffusionCoefficient D_kn_A;
    Modelica.Units.SI.DiffusionCoefficient D_kn_B;
    Modelica.Units.SI.DiffusionCoefficient D_AB;
  initial equation
    P_A = linspace(P_A_initial, P_A_initial, n);
    P_B = linspace(P_B_initial, P_B_initial, n);
  equation
// Energy balance
    port_b.h_outflow = (port_a.h_outflow*port_a.m_flow + port_h.Q_flow)/(-port_b.m_flow);
// Heat transfer
    port_h.Q_flow = -k*A*(T_gas - port_h.T);
// OBS: Double check later if it is negative or positive
// Ideal gas mixture
    P_A_bulk = port_a.p*port_a.Xi_outflow[1]/M_A/(port_a.Xi_outflow[1]/M_A + port_a.Xi_outflow[2]/M_B);
    P_B_bulk = port_a.p*port_a.Xi_outflow[2]/M_B/(port_a.Xi_outflow[1]/M_A + port_a.Xi_outflow[2]/M_B);
// Mass transport properties
    T_gas = Medium.temperature_phX(port_a.p, port_a.h_outflow, port_a.Xi_outflow);
    D_eff_A = ((epsilon/tau*D_kn_A)^(-1) + (epsilon/tau*D_AB)^(-1))^(-1);
    D_eff_B = ((epsilon/tau*D_kn_B)^(-1) + (epsilon/tau*D_AB)^(-1))^(-1);
    D_kn_A = 2*r_p/3*sqrt((8*Modelica.Constants.R*T_gas)/(Modelica.Constants.pi*M_A/1000));
    D_kn_B = 2*r_p/3*sqrt((8*Modelica.Constants.R*T_gas)/(Modelica.Constants.pi*M_B/1000));
    D_AB = 1.01325e-2*T_gas^1.75/sqrt(1/(M_A^(-1) + M_B^(-1)))/(port_a.p)/(V_A^(1/3) + V_B^(1/3))^2; // Assumes that pressure constant along the porous media
// Fick diffusion for A
    (delta/n)*der(P_A[1]) = -D_AB*(P_A[1] - P_A_bulk)/delta_ch + D_eff_A*(P_A[2] - P_A[1])/(delta/n);
    for i in 2:(n - 1) loop
      (delta/n)*der(P_A[i]) = -D_eff_A*(P_A[i] - P_A[i - 1])/(delta/n) + D_eff_A*(P_A[i + 1] - P_A[i])/(delta/n);
    end for;
    (delta/n)*der(P_A[end]) = -v_A*current/(n_e*Modelica.Constants.F*A)*Modelica.Constants.R*T_gas - D_eff_A*(P_A[end] - P_A[end - 1])/(delta/n);
// Fick diffusion for B
    (delta/n)*der(P_B[1]) = -D_AB*(P_B[1] - P_B_bulk)/delta_ch + D_eff_B*(P_B[2] - P_B[1])/(delta/n);
    for i in 2:(n - 1) loop
      (delta/n)*der(P_B[i]) = -D_eff_B*(P_B[i] - P_B[i - 1])/(delta/n) + D_eff_B*(P_B[i + 1] - P_B[i])/(delta/n);
    end for;
    (delta/n)*der(P_B[end]) = -v_B*current/(n_e*Modelica.Constants.F*A)*Modelica.Constants.R*T_gas - D_eff_B*(P_B[end] - P_B[end - 1])/(delta/n);
// Inlet mass port
    port_a.Xi_outflow = inStream(port_a.Xi_outflow);
    port_a.h_outflow = inStream(port_a.h_outflow);
// Outlet mass port
    port_b.m_flow = -(port_a.m_flow + D_AB*(P_A[1] - P_A_bulk)/delta_ch*A/(Modelica.Constants.R*T_gas)*M_A/1000 + D_AB*(P_B[1] - P_B_bulk)/delta_ch*A/(Modelica.Constants.R*T_gas)*M_B/1000);
    port_b.Xi_outflow[1] = (port_a.m_flow*port_a.Xi_outflow[1] + D_AB*(P_A[1] - P_A_bulk)/delta_ch*A/(Modelica.Constants.R*T_gas)*M_A/1000)/(-port_b.m_flow);
    port_b.Xi_outflow[2] = (port_a.m_flow*port_a.Xi_outflow[2] + D_AB*(P_B[1] - P_B_bulk)/delta_ch*A/(Modelica.Constants.R*T_gas)*M_B/1000)/(-port_b.m_flow);
// Pressure drop
    port_b.p = port_a.p;
// Outputs
    P_A_tpb = P_A[end];
    P_B_tpb = P_B[end];
// OBS: This model does not hold mass and enthalpy.
    annotation(
      Icon(graphics = {Rectangle(origin = {0, -30}, lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.CrossDiag, extent = {{-100, 10}, {100, -10}}), Rectangle(origin = {0, 10}, lineColor = {0, 0, 255}, extent = {{-100, 30}, {100, -30}})}));
  end porousDiffusion;

  model cellUnit "A cell segment with finite area"
    parameter Modelica.Units.SI.Area Area "Segment area";
    parameter Modelica.Units.SI.Length h_channel "Gas channel height";
    parameter Modelica.Units.SI.HeatCapacity Cp "Segment heat capacity";
    parameter Modelica.Units.SI.Temperature Tinit "Initial temperature";
    ocv nernst(A = Area) annotation(
      Placement(transformation(origin = {-10, 64}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    ohmic YSZ(delta = 12e-6 - 2e-6, sigma_0 = 3.6e7, theta = -1, E_act = 8e4, A = Area) annotation(
      Placement(transformation(origin = {-10, 2}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    ohmic GDC(delta = 10e-6 - 2e-6, sigma_0 = 1.09e7, theta = -1, E_act = 61.751e3, A = Area) annotation(
      Placement(transformation(origin = {-10, -46}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    ohmic interdiffusion(delta = 4e-6, sigma_0 = 1715, theta = 0, E_act = 8785*8.314510, A = Area) annotation(
      Placement(transformation(origin = {-10, -22}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    activation fuelElectrode(j_0_inf = 0.56*1.82527e6, theta = 1, a = -0.1, b = 0.33, m = 0, E_act = 105169.61134559404, alpha = 0.59, A = Area) annotation(
      Placement(transformation(origin = {-10, 30}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    activation airElectrode(j_0_inf = 1.51556e8, theta = 1, a = 0, b = 0, m = 0.22, E_act = 139904.5288542306, alpha = 0.65, A = Area) annotation(
      Placement(transformation(origin = {-10, -70}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    Modelica.Electrical.Analog.Sensors.CurrentSensor currentSensor annotation(
      Placement(transformation(origin = {-10, 90}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    Modelica.Thermal.HeatTransfer.Components.HeatCapacitor heatCapacitor(C = Cp, T(start = Tinit, fixed = true)) annotation(
      Placement(transformation(origin = {44, 64}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    Modelica.Electrical.Analog.Interfaces.PositivePin pin_p annotation(
      Placement(transformation(origin = {-10, 116}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-50, 100}, extent = {{-10, -10}, {10, 10}})));
    Modelica.Electrical.Analog.Interfaces.NegativePin pin_n annotation(
      Placement(transformation(origin = {-10, -100}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-52, -100}, extent = {{-10, -10}, {10, 10}})));
    Modelica.Fluid.Interfaces.FluidPort_a fuelIn(redeclare package Medium = nest.HydrogenWater) annotation(
      Placement(transformation(origin = {-160, 90}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-110, 50}, extent = {{-10, -10}, {10, 10}})));
    Modelica.Fluid.Interfaces.FluidPort_b fuelOut(redeclare package Medium = nest.HydrogenWater) annotation(
      Placement(transformation(origin = {-60, 90}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {110, 50}, extent = {{-10, -10}, {10, 10}})));
    Modelica.Fluid.Interfaces.FluidPort_a airIn(redeclare package Medium = Modelica.Media.IdealGases.MixtureGases.CombustionAir) annotation(
      Placement(transformation(origin = {-148, -110}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-110, -50}, extent = {{-10, -10}, {10, 10}})));
    Modelica.Fluid.Interfaces.FluidPort_b airOut(redeclare package Medium = Modelica.Media.IdealGases.MixtureGases.CombustionAir) annotation(
      Placement(transformation(origin = {-54, -110}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {110, -50}, extent = {{-10, -10}, {10, 10}})));
    Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_a port_a annotation(
      Placement(transformation(origin = {34, 140}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {48, 100}, extent = {{-10, -10}, {10, 10}})));
    nest.porousDiffusion airChannel(A = Area, M_A = 28.01340, M_B = 32, redeclare package Medium = Modelica.Media.IdealGases.MixtureGases.CombustionAir, P_A_initial = 0, P_B_initial = 1e5, V_A = 18.5, V_B = 16.3, delta = 3e-5, epsilon = 0.3, r_p = 3e-7, tau = 2.8, v_A = 0, v_B = 0.5, delta_ch = 0.0001, n = 5, k = 1) annotation(
      Placement(transformation(origin = {-78, -110}, extent = {{-10, 10}, {10, -10}})));
    nest.porousDiffusion fuelChannel(A = Area, redeclare package Medium = nest.HydrogenWater, P_A_initial = 5e4, P_B_initial = 5e4, delta_ch = 0.0001, epsilon = 0.3, k = 10, v_A = 1, v_B = -1, delta = 310e-6, tau = 3, r_p = 3e-7, M_A = 2.016, M_B = 18.02, V_A = 6.12, V_B = 13.1, n = 10) annotation(
      Placement(transformation(origin = {-90, 90}, extent = {{-10, -10}, {10, 10}})));
    Modelica.Electrical.Analog.Basic.Capacitor fuelCapacitor(C = 1.3*Area) annotation(
      Placement(transformation(origin = {10, 30}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    Modelica.Electrical.Analog.Basic.Capacitor airCapacitor(C = 100*Area) annotation(
      Placement(transformation(origin = {12, -70}, extent = {{-10, -10}, {10, 10}}, rotation = -90)));
    Modelica.Fluid.Vessels.ClosedVolume airChannelVol(nPorts = 2, redeclare package Medium = Modelica.Media.IdealGases.MixtureGases.CombustionAir, T_start = Tinit, V = Area*h_channel, X_start = {0, 1}, use_portsData = false) annotation(
      Placement(transformation(origin = {-120, -140}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
    Modelica.Fluid.Vessels.ClosedVolume fuelChannelVol(nPorts = 2, T_start = Tinit, use_portsData = false, V = Area*h_channel, X_start = {0.1, 0.9}, redeclare package Medium = nest.HydrogenWater) annotation(
      Placement(transformation(origin = {-130, 120}, extent = {{-10, -10}, {10, 10}})));
  equation
    connect(nernst.n, fuelElectrode.p) annotation(
      Line(points = {{-10, 54}, {-10, 40}}, color = {0, 0, 255}));
    connect(fuelElectrode.n, YSZ.p) annotation(
      Line(points = {{-10, 20}, {-10, 12}}, color = {0, 0, 255}));
    connect(YSZ.n, interdiffusion.p) annotation(
      Line(points = {{-10, -8}, {-10, -12}}, color = {0, 0, 255}));
    connect(interdiffusion.n, GDC.p) annotation(
      Line(points = {{-10, -32}, {-10, -36}}, color = {0, 0, 255}));
    connect(GDC.n, airElectrode.p) annotation(
      Line(points = {{-10, -56}, {-10, -60}}, color = {0, 0, 255}));
    connect(currentSensor.n, nernst.p) annotation(
      Line(points = {{-10, 80}, {-10, 74}}, color = {0, 0, 255}));
    connect(heatCapacitor.port, nernst.h) annotation(
      Line(points = {{34, 64}, {0, 64}}, color = {191, 0, 0}));
    connect(fuelElectrode.h, heatCapacitor.port) annotation(
      Line(points = {{0, 30}, {0, 12}, {34, 12}, {34, 64}}, color = {191, 0, 0}));
    connect(YSZ.h, heatCapacitor.port) annotation(
      Line(points = {{0, 2}, {34, 2}, {34, 64}}, color = {191, 0, 0}));
    connect(interdiffusion.h, heatCapacitor.port) annotation(
      Line(points = {{0, -22}, {34, -22}, {34, 64}}, color = {191, 0, 0}));
    connect(GDC.h, heatCapacitor.port) annotation(
      Line(points = {{0, -46}, {34, -46}, {34, 64}}, color = {191, 0, 0}));
    connect(port_a, heatCapacitor.port) annotation(
      Line(points = {{34, 140}, {34, 64}}, color = {191, 0, 0}));
    connect(fuelCapacitor.p, fuelElectrode.p) annotation(
      Line(points = {{10, 40}, {10, 48}, {-10, 48}, {-10, 40}}, color = {0, 0, 255}));
    connect(fuelCapacitor.n, fuelElectrode.n) annotation(
      Line(points = {{10, 20}, {-10, 20}}, color = {0, 0, 255}));
    connect(airElectrode.p, airCapacitor.p) annotation(
      Line(points = {{-10, -60}, {12, -60}}, color = {0, 0, 255}));
    connect(airCapacitor.n, airElectrode.n) annotation(
      Line(points = {{12, -80}, {-10, -80}}, color = {0, 0, 255}));
    connect(fuelOut, fuelChannel.port_b) annotation(
      Line(points = {{-60, 90}, {-80, 90}}));
    connect(airChannel.port_b, airOut) annotation(
      Line(points = {{-68, -110}, {-54, -110}}, color = {0, 127, 255}));
    connect(pin_p, currentSensor.p) annotation(
      Line(points = {{-10, 116}, {-10, 100}}, color = {0, 0, 255}));
    connect(heatCapacitor.port, fuelChannel.port_h) annotation(
      Line(points = {{34, 64}, {34, 130}, {-95, 130}, {-95, 100}}, color = {191, 0, 0}));
    connect(heatCapacitor.port, airChannel.port_h) annotation(
      Line(points = {{34, 64}, {34, -140}, {-83, -140}, {-83, -120}}, color = {191, 0, 0}));
    connect(fuelChannel.P_A_tpb, nernst.P_H2) annotation(
      Line(points = {{-95, 79}, {-95, 70}, {-22, 70}}, color = {0, 0, 127}));
    connect(fuelChannel.P_A_tpb, airElectrode.P_H2) annotation(
      Line(points = {{-95, 79}, {-95, -64}, {-22, -64}}, color = {0, 0, 127}));
    connect(airChannel.P_B_tpb, airElectrode.P_O2) annotation(
      Line(points = {{-73, -99}, {-73, -76}, {-22, -76}}, color = {0, 0, 127}));
    connect(fuelElectrode.P_O2, airChannel.P_B_tpb) annotation(
      Line(points = {{-22, 24}, {-73, 24}, {-73, -99}}, color = {0, 0, 127}));
    connect(nernst.P_O2, airChannel.P_B_tpb) annotation(
      Line(points = {{-22, 58}, {-73, 58}, {-73, -99}}, color = {0, 0, 127}));
    connect(airChannel.current, currentSensor.i) annotation(
      Line(points = {{-73, -122}, {-73, -132}, {-40, -132}, {-40, 90}, {-20, 90}}, color = {0, 0, 127}));
    connect(fuelChannel.current, currentSensor.i) annotation(
      Line(points = {{-85, 102}, {-85, 112}, {-40, 112}, {-40, 90}, {-20, 90}}, color = {0, 0, 127}));
    connect(fuelChannel.P_A_tpb, fuelElectrode.P_H2) annotation(
      Line(points = {{-95, 79}, {-95, 36}, {-22, 36}}, color = {0, 0, 127}));
    connect(fuelChannel.P_B_tpb, nernst.P_H2O) annotation(
      Line(points = {{-85, 79}, {-85, 64}, {-22, 64}}, color = {0, 0, 127}));
    connect(fuelChannel.P_B_tpb, fuelElectrode.P_H2O) annotation(
      Line(points = {{-85, 79}, {-85, 30}, {-22, 30}}, color = {0, 0, 127}));
    connect(fuelChannel.P_B_tpb, airElectrode.P_H2O) annotation(
      Line(points = {{-85, 79}, {-85, -70}, {-22, -70}}, color = {0, 0, 127}));
    connect(airElectrode.h, heatCapacitor.port) annotation(
      Line(points = {{0, -70}, {0, -86}, {34, -86}, {34, 64}}, color = {191, 0, 0}));
    connect(airIn, airChannelVol.ports[1]) annotation(
      Line(points = {{-148, -110}, {-120, -110}, {-120, -130}}));
    connect(airChannel.port_a, airChannelVol.ports[2]) annotation(
      Line(points = {{-88, -110}, {-120, -110}, {-120, -130}}, color = {0, 127, 255}));
    connect(fuelIn, fuelChannelVol.ports[1]) annotation(
      Line(points = {{-160, 90}, {-130, 90}, {-130, 110}}));
    connect(fuelChannel.port_a, fuelChannelVol.ports[2]) annotation(
      Line(points = {{-100, 90}, {-130, 90}, {-130, 110}}, color = {0, 127, 255}));
  connect(airElectrode.n, pin_n) annotation(
      Line(points = {{-10, -80}, {-10, -100}}, color = {0, 0, 255}));
    annotation(
      Diagram(coordinateSystem(extent = {{-180, 160}, {60, -160}})),
      version = "",
      uses(Modelica(version = "4.0.0")),
  Icon(graphics = {Rectangle(origin = {0, 70}, lineColor = {0, 0, 255}, extent = {{-100, 30}, {100, -30}}), Rectangle( origin = {0, -70},lineColor = {0, 0, 255}, extent = {{-100, 30}, {100, -30}}), Rectangle( lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.CrossDiag, extent = {{-100, 40}, {100, -40}})}));
  end cellUnit;
  
  model cell1D "Multiple finite element segments represeting a cell (Plug-flow reactor)"
    parameter Modelica.Units.SI.Area ActiveArea "Cell active area";
    parameter Modelica.Units.SI.Length h_channel "Gas channel height";
    parameter Modelica.Units.SI.HeatCapacity Cp "Segment heat capacity";
    parameter Modelica.Units.SI.Temperature Tinit "Initial temperature";
    constant Integer segments = 10;
    cellUnit cellUnit1(Area = ActiveArea/segments, Tinit= Tinit, Cp = Cp/segments, h_channel = h_channel) annotation(
      Placement(transformation(origin = {-10, 10}, extent = {{-10, -10}, {10, 10}})));
    cellUnit cellUnit2(Area = ActiveArea/segments, Tinit= Tinit, Cp = Cp/segments, h_channel = h_channel) annotation(
      Placement(transformation(origin = {30, 10}, extent = {{-10, -10}, {10, 10}})));
    cellUnit cellUnit3(Area = ActiveArea/segments, Tinit= Tinit, Cp = Cp/segments, h_channel = h_channel) annotation(
      Placement(transformation(origin = {70, 10}, extent = {{-10, -10}, {10, 10}})));
    cellUnit cellUnit4(Area = ActiveArea/segments, Tinit= Tinit, Cp = Cp/segments, h_channel = h_channel) annotation(
      Placement(transformation(origin = {110, 10}, extent = {{-10, -10}, {10, 10}})));
    cellUnit cellUnit5(Area = ActiveArea/segments, Tinit= Tinit, Cp = Cp/segments, h_channel = h_channel) annotation(
      Placement(transformation(origin = {150, 10}, extent = {{-10, -10}, {10, 10}})));
    cellUnit cellUnit10(Area = ActiveArea/segments, Tinit= Tinit, Cp = Cp/segments, h_channel = h_channel) annotation(
      Placement(transformation(origin = {350, 10}, extent = {{-10, -10}, {10, 10}})));
    cellUnit cellUnit9(Area = ActiveArea/segments, Tinit= Tinit, Cp = Cp/segments, h_channel = h_channel) annotation(
      Placement(transformation(origin = {310, 10}, extent = {{-10, -10}, {10, 10}})));
    cellUnit cellUnit8(Area = ActiveArea/segments, Tinit= Tinit, Cp = Cp/segments, h_channel = h_channel) annotation(
      Placement(transformation(origin = {270, 10}, extent = {{-10, -10}, {10, 10}})));
    cellUnit cellUnit7(Area = ActiveArea/segments, Tinit= Tinit, Cp = Cp/segments, h_channel = h_channel) annotation(
      Placement(transformation(origin = {230, 10}, extent = {{-10, -10}, {10, 10}})));
    cellUnit cellUnit6(Area = ActiveArea/segments, Tinit= Tinit, Cp = Cp/segments, h_channel = h_channel) annotation(
      Placement(transformation(origin = {190, 10}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Fluid.Interfaces.FluidPort_a fuelIn(redeclare package Medium = nest.HydrogenWater) annotation(
      Placement(transformation(origin = {-58, 16}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-110, 50}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Fluid.Interfaces.FluidPort_b fuelOut(redeclare package Medium = nest.HydrogenWater) annotation(
      Placement(transformation(origin = {380, 16}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {110, 50}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Fluid.Interfaces.FluidPort_a airIn(redeclare package Medium = Modelica.Media.IdealGases.MixtureGases.CombustionAir) annotation(
      Placement(transformation(origin = {-70, 6}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-110, -50}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Fluid.Interfaces.FluidPort_b airOut(redeclare package Medium = Modelica.Media.IdealGases.MixtureGases.CombustionAir) annotation(
      Placement(transformation(origin = {406, 6}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {110, -50}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Thermal.HeatTransfer.Interfaces.HeatPort_a port_a annotation(
      Placement(transformation(origin = {166, 78}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {48, 100}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Interfaces.NegativePin pin_n annotation(
      Placement(transformation(origin = {168, -24}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-52, -100}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Interfaces.PositivePin pin_p annotation(
      Placement(transformation(origin = {168, 50}, extent = {{-10, -10}, {10, 10}}), iconTransformation(origin = {-50, 100}, extent = {{-10, -10}, {10, 10}})));
  equation
    connect(fuelIn, cellUnit1.fuelIn) annotation(
      Line(points = {{-58, 16}, {-20, 16}}));
  connect(cellUnit1.airIn, airIn) annotation(
      Line(points = {{-20, 6}, {-70, 6}}, color = {0, 127, 255}));
  connect(cellUnit1.fuelOut, cellUnit2.fuelIn) annotation(
      Line(points = {{2, 16}, {20, 16}}, color = {0, 127, 255}));
  connect(cellUnit2.fuelOut, cellUnit3.fuelIn) annotation(
      Line(points = {{42, 16}, {60, 16}}, color = {0, 127, 255}));
  connect(cellUnit3.fuelOut, cellUnit4.fuelIn) annotation(
      Line(points = {{82, 16}, {100, 16}}, color = {0, 127, 255}));
  connect(cellUnit4.fuelOut, cellUnit5.fuelIn) annotation(
      Line(points = {{122, 16}, {140, 16}}, color = {0, 127, 255}));
  connect(cellUnit5.fuelOut, cellUnit6.fuelIn) annotation(
      Line(points = {{162, 16}, {180, 16}}, color = {0, 127, 255}));
  connect(cellUnit6.fuelOut, cellUnit7.fuelIn) annotation(
      Line(points = {{202, 16}, {220, 16}}, color = {0, 127, 255}));
  connect(cellUnit7.fuelOut, cellUnit8.fuelIn) annotation(
      Line(points = {{242, 16}, {260, 16}}, color = {0, 127, 255}));
  connect(cellUnit8.fuelOut, cellUnit9.fuelIn) annotation(
      Line(points = {{282, 16}, {300, 16}}, color = {0, 127, 255}));
  connect(cellUnit9.fuelOut, cellUnit10.fuelIn) annotation(
      Line(points = {{322, 16}, {340, 16}}, color = {0, 127, 255}));
  connect(cellUnit10.fuelOut, fuelOut) annotation(
      Line(points = {{362, 16}, {380, 16}}, color = {0, 127, 255}));
  connect(cellUnit1.airOut, cellUnit2.airIn) annotation(
      Line(points = {{2, 6}, {20, 6}}, color = {0, 127, 255}));
  connect(cellUnit2.airOut, cellUnit3.airIn) annotation(
      Line(points = {{42, 6}, {60, 6}}, color = {0, 127, 255}));
  connect(cellUnit3.airOut, cellUnit4.airIn) annotation(
      Line(points = {{82, 6}, {100, 6}}, color = {0, 127, 255}));
  connect(cellUnit4.airOut, cellUnit5.airIn) annotation(
      Line(points = {{122, 6}, {140, 6}}, color = {0, 127, 255}));
  connect(cellUnit5.airOut, cellUnit6.airIn) annotation(
      Line(points = {{162, 6}, {180, 6}}, color = {0, 127, 255}));
  connect(cellUnit6.airOut, cellUnit7.airIn) annotation(
      Line(points = {{202, 6}, {220, 6}}, color = {0, 127, 255}));
  connect(cellUnit7.airOut, cellUnit8.airIn) annotation(
      Line(points = {{242, 6}, {260, 6}}, color = {0, 127, 255}));
  connect(cellUnit8.airOut, cellUnit9.airIn) annotation(
      Line(points = {{282, 6}, {300, 6}}, color = {0, 127, 255}));
  connect(cellUnit9.airOut, cellUnit10.airIn) annotation(
      Line(points = {{322, 6}, {340, 6}}, color = {0, 127, 255}));
  connect(cellUnit10.airOut, airOut) annotation(
      Line(points = {{362, 6}, {406, 6}}, color = {0, 127, 255}));
  connect(cellUnit1.pin_n, pin_n) annotation(
      Line(points = {{-16, 0}, {-16, -24}, {168, -24}}, color = {0, 0, 255}));
  connect(cellUnit2.pin_n, pin_n) annotation(
      Line(points = {{24, 0}, {24, -24}, {168, -24}}, color = {0, 0, 255}));
  connect(cellUnit3.pin_n, pin_n) annotation(
      Line(points = {{64, 0}, {64, -24}, {168, -24}}, color = {0, 0, 255}));
  connect(cellUnit4.pin_n, pin_n) annotation(
      Line(points = {{104, 0}, {104, -24}, {168, -24}}, color = {0, 0, 255}));
  connect(cellUnit5.pin_n, pin_n) annotation(
      Line(points = {{144, 0}, {144, -24}, {168, -24}}, color = {0, 0, 255}));
  connect(cellUnit6.pin_n, pin_n) annotation(
      Line(points = {{184, 0}, {186, 0}, {186, -24}, {168, -24}}, color = {0, 0, 255}));
  connect(cellUnit7.pin_n, pin_n) annotation(
      Line(points = {{224, 0}, {224, -24}, {168, -24}}, color = {0, 0, 255}));
  connect(cellUnit8.pin_n, pin_n) annotation(
      Line(points = {{264, 0}, {264, -24}, {168, -24}}, color = {0, 0, 255}));
  connect(cellUnit9.pin_n, pin_n) annotation(
      Line(points = {{304, 0}, {304, -24}, {168, -24}}, color = {0, 0, 255}));
  connect(cellUnit10.pin_n, pin_n) annotation(
      Line(points = {{344, 0}, {344, -24}, {168, -24}}, color = {0, 0, 255}));
  connect(cellUnit1.pin_p, pin_p) annotation(
      Line(points = {{-14, 20}, {-16, 20}, {-16, 50}, {168, 50}}, color = {0, 0, 255}));
  connect(cellUnit2.pin_p, pin_p) annotation(
      Line(points = {{26, 20}, {24, 20}, {24, 50}, {168, 50}}, color = {0, 0, 255}));
  connect(cellUnit3.pin_p, pin_p) annotation(
      Line(points = {{66, 20}, {64, 20}, {64, 50}, {168, 50}}, color = {0, 0, 255}));
  connect(cellUnit4.pin_p, pin_p) annotation(
      Line(points = {{106, 20}, {104, 20}, {104, 50}, {168, 50}}, color = {0, 0, 255}));
  connect(cellUnit5.pin_p, pin_p) annotation(
      Line(points = {{146, 20}, {146, 50}, {168, 50}}, color = {0, 0, 255}));
  connect(cellUnit6.pin_p, pin_p) annotation(
      Line(points = {{186, 20}, {184, 20}, {184, 50}, {168, 50}}, color = {0, 0, 255}));
  connect(cellUnit7.pin_p, pin_p) annotation(
      Line(points = {{226, 20}, {224, 20}, {224, 50}, {168, 50}}, color = {0, 0, 255}));
  connect(cellUnit8.pin_p, pin_p) annotation(
      Line(points = {{266, 20}, {264, 20}, {264, 50}, {168, 50}}, color = {0, 0, 255}));
  connect(cellUnit9.pin_p, pin_p) annotation(
      Line(points = {{306, 20}, {304, 20}, {304, 50}, {168, 50}}, color = {0, 0, 255}));
  connect(cellUnit10.pin_p, pin_p) annotation(
      Line(points = {{346, 20}, {344, 20}, {344, 50}, {168, 50}}, color = {0, 0, 255}));
  connect(cellUnit1.port_a, port_a) annotation(
      Line(points = {{-6, 20}, {-6, 78}, {166, 78}}, color = {191, 0, 0}));
  connect(cellUnit2.port_a, port_a) annotation(
      Line(points = {{34, 20}, {34, 78}, {166, 78}}, color = {191, 0, 0}));
  connect(cellUnit3.port_a, port_a) annotation(
      Line(points = {{74, 20}, {74, 78}, {166, 78}}, color = {191, 0, 0}));
  connect(cellUnit4.port_a, port_a) annotation(
      Line(points = {{114, 20}, {112, 20}, {112, 78}, {166, 78}}, color = {191, 0, 0}));
  connect(cellUnit5.port_a, port_a) annotation(
      Line(points = {{154, 20}, {156, 20}, {156, 78}, {166, 78}}, color = {191, 0, 0}));
  connect(cellUnit6.port_a, port_a) annotation(
      Line(points = {{194, 20}, {194, 78}, {166, 78}}, color = {191, 0, 0}));
  connect(cellUnit7.port_a, port_a) annotation(
      Line(points = {{234, 20}, {234, 78}, {166, 78}}, color = {191, 0, 0}));
  connect(cellUnit8.port_a, port_a) annotation(
      Line(points = {{274, 20}, {274, 78}, {166, 78}}, color = {191, 0, 0}));
  connect(cellUnit9.port_a, port_a) annotation(
      Line(points = {{314, 20}, {312, 20}, {312, 78}, {166, 78}}, color = {191, 0, 0}));
  connect(cellUnit10.port_a, port_a) annotation(
      Line(points = {{354, 20}, {354, 78}, {166, 78}}, color = {191, 0, 0}));
    annotation(
      Diagram(coordinateSystem(extent = {{-80, 100}, {420, -40}})),
  Icon(graphics = {Rectangle(lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.CrossDiag, extent = {{-100, 40}, {-80, -40}}), Rectangle(origin = {30, 0}, lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.CrossDiag, extent = {{-100, 40}, {-80, -40}}), Rectangle(origin = {60, 0},lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.CrossDiag, extent = {{-100, 40}, {-80, -40}}), Rectangle(origin = {90, 0},lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.CrossDiag, extent = {{-100, 40}, {-80, -40}}), Rectangle(origin = {120, 0}, lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.CrossDiag, extent = {{-100, 40}, {-80, -40}}), Rectangle(origin = {180, 0}, lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.CrossDiag, extent = {{-100, 40}, {-80, -40}}), Rectangle(origin = {150, 0}, lineColor = {0, 0, 255}, fillColor = {255, 255, 255}, fillPattern = FillPattern.CrossDiag, extent = {{-100, 40}, {-80, -40}}), Rectangle(origin = {0, 70}, lineColor = {0, 0, 255}, extent = {{-100, 30}, {100, -30}}), Rectangle(origin = {0, -70}, lineColor = {0, 0, 255}, extent = {{-100, 30}, {100, -30}})}));
  end cell1D;
  
  model example
    Modelica.Fluid.Sources.FixedBoundary fuelOut(redeclare package Medium = HydrogenWater, T(displayUnit = "K") = 1033, X = {0.1, 0.9}, nPorts = 1, p = 1e5) annotation(
      Placement(transformation(origin = {70, 30}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
    Modelica.Fluid.Sources.FixedBoundary airOut(redeclare package Medium = Modelica.Media.IdealGases.MixtureGases.CombustionAir, T(displayUnit = "K") = 1033, X = {0, 1}, nPorts = 1, p = 1e5, use_T = true, use_p = true) annotation(
      Placement(transformation(origin = {70, -10}, extent = {{-10, -10}, {10, 10}}, rotation = 180)));
  Modelica.Fluid.Sources.MassFlowSource_T fuelIn(redeclare package Medium = HydrogenWater, T(displayUnit = "degC") = 1123.15, X = {0.1, 0.9}, m_flow = 0.029354*10, nPorts = 1) annotation(
      Placement(transformation(origin = {-30, 30}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Fluid.Sources.MassFlowSource_T airIn(redeclare package Medium = Modelica.Media.IdealGases.MixtureGases.CombustionAir, T(displayUnit = "degC") = 1123.15, X = {0, 1}, m_flow = 0.19570*10, nPorts = 1) annotation(
      Placement(transformation(origin = {-30, -10}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Basic.Ground ground annotation(
      Placement(transformation(origin = {-50, -56}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Thermal.HeatTransfer.Sources.FixedTemperature fixedTemperature(T(displayUnit = "degC") = 930.15) annotation(
      Placement(transformation(origin = {-90, 90}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Thermal.HeatTransfer.Components.ThermalResistor thermalResistor(R = 1e-8) annotation(
      Placement(transformation(origin = {-30, 90}, extent = {{-10, -10}, {10, 10}})));
    inner Modelica.Fluid.System system annotation(
      Placement(transformation(origin = {70, 70}, extent = {{-10, -10}, {10, 10}})));
  Modelica.Electrical.Analog.Sources.SignalVoltage signalVoltage annotation(
      Placement(transformation(origin = {-50, 10}, extent = {{-10, -10}, {10, 10}}, rotation = 90)));
  Modelica.Blocks.Sources.Constant const(k = 1.25) annotation(
      Placement(transformation(origin = {-90, 10}, extent = {{-10, -10}, {10, 10}})));
  cell1D cell1D1(h_channel = 0.001, Cp = 1, Tinit = 930.15, ActiveArea = 32*5)  annotation(
      Placement(transformation(origin = {22, 10}, extent = {{-10, -10}, {10, 10}})));
  equation
    connect(const.y, signalVoltage.v) annotation(
      Line(points = {{-79, 10}, {-62, 10}}, color = {0, 0, 127}));
  connect(signalVoltage.n, cell1D1.pin_p) annotation(
      Line(points = {{-50, 20}, {-50, 60}, {17, 60}, {17, 20}}, color = {0, 0, 255}));
  connect(cell1D1.pin_n, signalVoltage.p) annotation(
      Line(points = {{17, 0}, {17, -40}, {-50, -40}, {-50, 0}}, color = {0, 0, 255}));
  connect(ground.p, signalVoltage.p) annotation(
      Line(points = {{-50, -46}, {-50, 0}}, color = {0, 0, 255}));
  connect(cell1D1.airOut, airOut.ports[1]) annotation(
      Line(points = {{34, 6}, {40, 6}, {40, -10}, {60, -10}}, color = {0, 127, 255}));
  connect(cell1D1.airIn, airIn.ports[1]) annotation(
      Line(points = {{12, 6}, {0, 6}, {0, -10}, {-20, -10}}, color = {0, 127, 255}));
  connect(cell1D1.fuelIn, fuelIn.ports[1]) annotation(
      Line(points = {{12, 16}, {0, 16}, {0, 30}, {-20, 30}}, color = {0, 127, 255}));
  connect(fixedTemperature.port, thermalResistor.port_a) annotation(
      Line(points = {{-80, 90}, {-40, 90}}, color = {191, 0, 0}));
  connect(thermalResistor.port_b, cell1D1.port_a) annotation(
      Line(points = {{-20, 90}, {26, 90}, {26, 20}}, color = {191, 0, 0}));
  connect(cell1D1.fuelOut, fuelOut.ports[1]) annotation(
      Line(points = {{34, 16}, {42, 16}, {42, 30}, {60, 30}}, color = {0, 127, 255}));
    annotation(
      Diagram(coordinateSystem(extent = {{-100, 100}, {80, -80}})));
  end example;
  annotation(
    uses(Modelica(version = "4.0.0")));
end nest;
