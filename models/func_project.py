import pybamm   # type: ignore
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# import random



#############
# utilities #
#############

def print_all_variables(parameter_values, file_name):
    """
    Sorts the parameter dictionary alphabetically by keys and writes it to a text file.

    Args:
        parameter_values (dict): Dictionary of parameter values.
        file_name (str): Name of the output text file.
    """
    # Open the file in write mode
    with open(file_name, "w") as f:
        # Sort the dictionary by keys
        sorted_items = sorted(parameter_values.items())
        
        # Write each key-value pair to the file
        for key, value in sorted_items:
            f.write(f"{key}\t{value}\n")  # Tab-separated for easy copying into Excel

    print(f"Parameters have been written to {file_name} in sorted order.")



def update_parameter_set(old_set):
    '''
    updating the old set with thermal properties from ORegan2022,
    return a new parameter set

    the parameters are written into "parameter values.txt"
    '''
    for param in old_set:
        try:
            if old_set[param] != pybamm.ParameterValues("ORegan2022")[param]:
                old_set.update({param: pybamm.ParameterValues("ORegan2022")[param]})
            else:
                continue
        except:
            #print("There is no value for this parameter in the new set")
            continue

    
        # create a copy of the parameter and updates with values for e- limited SEI model
    parameters=old_set.copy()
    parameters.update(      # update the parameter set with information for electron tunneling related mechanisms
        {
            'Initial SEI thickness [m]': 5e-9,
            'SEI lithium ion conductivity [S.m-1]': 1.0e-7,
            'Tunneling barrier factor [m-1]': 7e-9,
            'Tunneling distance for electrons [m]': 1.4e-9,
            "SEI solvent diffusivity [m2.s-1]": 2.5e-22,
            "Lithium plating kinetic rate constant [m.s-1]": 1e-09,
            "Dead lithium decay constant [s-1]": 1e-06,

        }, check_already_exists=False)

    # write the parameter values and model variables to a text file
    f = open("parameter_values.txt", "w")
    f.write(f"{parameters}")
    f.close()

    # returns the completed parameter set
    return parameters



def plot_results(solution_set, variables):
    '''
    plot graph of SEI thickness, Capacity loss and Voltage against total throughput

        solution_set = list of solutions after solving multiple models
        variables = variables changed during parametric sweep
    
    '''

    fig, axes = plt.subplots(3,2, layout="constrained", figsize=(10,12), gridspec_kw={"hspace": 0.1}, sharex=False)

    for index, sol in enumerate(solution_set):
        # independent variable
        Q_tot = sol["Throughput capacity [A.h]"].entries
        t = sol["Time [h]"].entries

        # various dependent variables
        Q_SEI = sol["Loss of capacity to negative SEI [A.h]"].entries
        J_SEI = sol["X-averaged negative electrode SEI interfacial current density [A.m-2]"].entries
        V_i = sol["Voltage [V]"].entries
        # electrolyte_V = sol["Electrolyte potential [V]"].entries
        # C_dis = sol["Discharge capacity [A.h]"].entries
        Q_LLI = sol["Total lithium lost [mol]"].entries #*96485.3 / 3600
        
        L_SEI = (sol["X-averaged negative total SEI thickness [m]"].entries) *1e9
        L_min = np.min(sol["X-averaged negative total SEI thickness [m]"].entries) *1e9
        delta_L = (L_SEI - L_min)

        # plot graph on the go

        axes[0][0].plot(Q_tot, Q_SEI, label=f"{variables[index]}", linestyle="--")
        axes[0][1].plot(Q_tot, J_SEI, label=f"{variables[index]}", linestyle="--")
        axes[1][0].plot(Q_tot, L_SEI, label=f"{variables[index]}", linestyle="--")
        axes[1][1].plot(t, delta_L, label=f"{variables[index]}", linestyle="--")
        axes[2][0].plot(t, V_i, label=f"{variables[index]}", linestyle="--")
        axes[2][1].plot(Q_tot, Q_LLI, label=f"{variables[index]}", linestyle="--")
        
    # graph formatting
    axes[0][0].set_xlabel("Total throughput capacity [A.h]")
    axes[0][0].set_ylabel("Loss of capacity to negative SEI [A.h]") 
    axes[0][0].legend()
    
    axes[0][1].set_xlabel("Total throughput capacity [A.h]")
    axes[0][1].set_ylabel("SEI interfacial current density [A.m-2]")

    axes[1][0].set_xlabel("Total throughput capacity [A.h]")
    axes[1][0].set_ylabel("Total SEI thickness [nm]")


    axes[1][1].set_xlabel("Time [h]")
    axes[1][1].set_ylabel("Net SEI growth [nm]")

    axes[2][0].set_xlabel("Total throughput capacity [A.h]")
    axes[2][0].set_ylabel("Voltage [V]")

    axes[2][1].set_xlabel("Total throughput capacity [A.h]")
    axes[2][1].set_ylabel("Loss of lithium inventory [A.h]")
    


def plot_capacity_loss(solution, model_name):
    '''
    Calculating the capacity lost as a percentage of the original full health capacity
    '''
    cap = solution["Discharge capacity [A.h]"].entries
    max_cap = np.max(cap)
    loss = max_cap - cap
    t = solution["Time [h]"].entries
    Q_tot = ["Throughput capacity [A.h]"].entries

    plt.figure()
    plt.plot(Q_tot, loss, label=f"{model_name}", layout="constrained", gridspec_kw={"hspace": 0.1}, sharex=False)
    plt.xlabel("Throughput capacity [A.h]")
    plt.ylabel("Capacity lost [%]")
    plt.legend()
    plt.show()



###################################################
# function parameters for composite electrode model
###################################################


def graphite_plating_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Defining temperature/concentration dependent parameters for plating
    Exchange-current density for Li plating reaction [A.m-2].

    References:
    ----------
    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.

    Parameters:
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    
    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter(
        "Primary: Lithium plating kinetic rate constant [m.s-1]"
    )

    return pybamm.constants.F * k_plating * c_e


def graphite_stripping_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Exchange-current density for Li stripping reaction [A.m-2].
    
    References
    ----------
    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.
    
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    
    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter(
        "Primary: Lithium plating kinetic rate constant [m.s-1]"
    )

    return pybamm.constants.F * k_plating * c_Li


def graphite_SEI_limited_dead_lithium_OKane2022(L_sei):
    """
    Decay rate for dead lithium formation [s-1].
    
    References
    ----------
    .. [1] Simon E. J. O'Kane, Weilong Ai, Ganesh Madabattula, Diega Alonso-Alvarez,
    Robert Timms, Valentin Sulzer, Jaqueline Sophie Edge, Billy Wu, Gregory J. Offer
    and Monica Marinescu. "Lithium-ion battery degradation: how to model it."
    Physical Chemistry: Chemical Physics 24, no. 13 (2022): 7909-7922.
    Parameters
    ----------
    L_sei : :class:`pybamm.Symbol`
        Total SEI thickness [m]
    
    Returns
    -------
    :class:`pybamm.Symbol`
        Dead lithium decay rate [s-1]
    """

    gamma_0 = pybamm.Parameter("Primary: Dead lithium decay constant [s-1]")
    L_inner_0 = pybamm.Parameter("Primary: Initial inner SEI thickness [m]")
    L_outer_0 = pybamm.Parameter("Primary: Initial outer SEI thickness [m]")
    L_sei_0 = L_inner_0 + L_outer_0

    gamma = gamma_0 * L_sei_0 / L_sei

    return gamma


def silicon_plating_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Exchange-current density for Li plating reaction [A.m-2].
    
    References
    ----------
    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.
    
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    
    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter(
        "Secondary: Lithium plating kinetic rate constant [m.s-1]"
    )

    return pybamm.constants.F * k_plating * c_e


def silicon_stripping_exchange_current_density_OKane2020(c_e, c_Li, T):
    """
    Exchange-current density for Li stripping reaction [A.m-2].
    
    References
    ----------
    .. [1] O’Kane, Simon EJ, Ian D. Campbell, Mohamed WJ Marzook, Gregory J. Offer, and
    Monica Marinescu. "Physical origin of the differential voltage minimum associated
    with lithium plating in Li-ion batteries." Journal of The Electrochemical Society
    167, no. 9 (2020): 090540.
    
    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_Li : :class:`pybamm.Symbol`
        Plated lithium concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]
    
    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """

    k_plating = pybamm.Parameter(
        "Secondary: Lithium plating kinetic rate constant [m.s-1]"
    )

    return pybamm.constants.F * k_plating * c_Li


def silicon_SEI_limited_dead_lithium_OKane2022(L_sei):
    """
    Decay rate for dead lithium formation [s-1].
    
    References
    ----------
    .. [1] Simon E. J. O'Kane, Weilong Ai, Ganesh Madabattula, Diega Alonso-Alvarez,
    Robert Timms, Valentin Sulzer, Jaqueline Sophie Edge, Billy Wu, Gregory J. Offer
    and Monica Marinescu. "Lithium-ion battery degradation: how to model it."
    Physical Chemistry: Chemical Physics 24, no. 13 (2022): 7909-7922.
    
    Parameters
    ----------
    L_sei : :class:`pybamm.Symbol`
        Total SEI thickness [m]
    
    Returns
    -------
    :class:`pybamm.Symbol`
        Dead lithium decay rate [s-1]
    """

    gamma_0 = pybamm.Parameter("Secondary: Dead lithium decay constant [s-1]")
    L_inner_0 = pybamm.Parameter("Secondary: Initial inner SEI thickness [m]")
    L_outer_0 = pybamm.Parameter("Secondary: Initial outer SEI thickness [m]")
    L_sei_0 = L_inner_0 + L_outer_0

    gamma = gamma_0 * L_sei_0 / L_sei

    return gamma


def graphite_LGM50_electrolyte_exchange_current_density_Chen2020(
    c_e, c_s_surf, c_s_max, T
):
    """
    Exchange-current density for Butler-Volmer reactions between graphite and LiPF6 in
    EC:DMC.

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    c_e : :class:`pybamm.Symbol`
        Electrolyte concentration [mol.m-3]
    c_s_surf : :class:`pybamm.Symbol`
        Particle concentration [mol.m-3]
    c_s_max : :class:`pybamm.Symbol`
        Maximum particle concentration [mol.m-3]
    T : :class:`pybamm.Symbol`
        Temperature [K]

    Returns
    -------
    :class:`pybamm.Symbol`
        Exchange-current density [A.m-2]
    """
    m_ref = 6.48e-7  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 35000
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5


def nmc_LGM50_ocp_Chen2020(sto):
    """
    LG M50 NMC open-circuit potential as a function of stoichiometry, fit taken
    from [1].

    References
    ----------
    .. [1] Chang-Hui Chen, Ferran Brosa Planella, Kieran O’Regan, Dominika Gastol, W.
    Dhammika Widanage, and Emma Kendrick. "Development of Experimental Techniques for
    Parameterization of Multi-scale Lithium-ion Battery Models." Journal of the
    Electrochemical Society 167 (2020): 080534.

    Parameters
    ----------
    sto: :class:`pybamm.Symbol`
        Electrode stoichiometry

    Returns
    -------
    :class:`pybamm.Symbol`
        Open-circuit potential
    """

    u_eq = (
        -0.8090 * sto
        + 4.4875
        - 0.0428 * np.tanh(18.5138 * (sto - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (sto - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (sto - 0.3120))
    )

    return u_eq