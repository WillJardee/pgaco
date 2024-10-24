import numpy as np
import matplotlib.pyplot as plt
import tqdm

from ACA import ACA_TSP
from PGACA import PolicyGradientACA 
from minmaxACA import ACA_minmax


###############################################################################
###############################################################################
###############################################################################
def stat_run():
    """Quick function for grabbing mean and std over ACA runs."""
    size = 100
    runs = 5 
    iterations = 500
    distance_matrix = np.random.randint(1, 10, size**2).reshape((size, size))
    # distance_matrix[np.where(distance_matrix == 0)] = 1e13

    def cal_total_distance(routine):
        return sum([distance_matrix[routine[i % size], routine[(i + 1) % size]]
                    for i in range(size)])

    

    print("Running ACA")

    from multiprocessing.dummy import Pool
    ACA_runs = []

    def run_ACA(throwaway):
        aca = ACA_TSP(func=cal_total_distance,
                      distance_matrix=distance_matrix,
                      params={"iterations": iterations})
        skaco_sol, skaco_cost = aca.run(metric = ["branching_factor"])
        return skaco_cost

    with Pool() as pool:
        results = pool.imap_unordered(run_ACA, [[]]*runs)
        for test in results:
            ACA_runs.append(test)
            # print(f"{skaco_sol}: {skaco_cost}")

    print("Running minmaxACA")
    mmACA_runs = []

    def run_mmACA(throwaway):
        mmaca = ACA_minmax(func=cal_total_distance,
                      distance_matrix=distance_matrix,
                      params={"iterations": iterations})
        skaco_sol, skaco_cost = mmaca.run(metric = ["branching_factor"])
        return skaco_cost

    with Pool() as pool:
        results = pool.imap_unordered(run_ACA, [[]]*runs)
        for test in results:
            mmACA_runs.append(test)
            # print(f"{skaco_sol}: {skaco_cost}")

    print("Running PGACA; fb")
    PGACAfb_runs = []
    for _ in range(runs):
        aca = PolicyGradientACA(func=cal_total_distance,
                                distance_matrix=distance_matrix,
                                params={
                                    "learning_rate": float(1)**(2),
                                    "gamma": 1,
                                    "advantage_type": "forward_backward",
                                    "iterations": iterations
                                })
        skaco_sol, skaco_cost = aca.run(metric = ["branching_factor"])
        PGACAfb_runs.append(skaco_cost)
        # print(f"{skaco_cost}")

    # print("Running clip-PGACA")
    # CPGACA_runs = []
    # for _ in range(runs):
    #     aca = ClipACA(func=cal_total_distance,
    #                             distance_matrix=distance_matrix,
    #                             params={
    #                             "learning_rate": float(1)**(2),
    #                             "gamma": 1,
    #                             "advantage_type": "forward_backward",
    #                             "iterations": iterations
    #                             })
    #     skaco_sol, skaco_cost = aca.run()
    #     CPGACA_runs.append(skaco_cost)
    #     # print(f"{skaco_cost}")

    # print("Running GDACA")
    # GDACA_runs = []
    # for _ in range(runs):
    #     aca = GDACA(func=cal_total_distance,
    #                 distance_matrix=distance_matrix,
    #                 params={
    #                     "iterations": iterations
    #                 })
    #     skaco_sol, skaco_cost = aca.run(metric = ["branching_factor"])
    #     GDACA_runs.append(skaco_cost)
    #     # print(f"{skaco_cost}")
    print("Running PGACA; baseline")
    PGACAb_runs = []
    for _ in range(runs):
        aca = PolicyGradientACA(func=cal_total_distance,
                                distance_matrix=distance_matrix,
                                params={
                                    "learning_rate": float(1)**(2),
                                    "gamma": 1,
                                    "advantage_type": "baseline",
                                    "iterations": iterations
                                })
        skaco_sol, skaco_cost = aca.run(metric = ["branching_factor"])
        PGACAb_runs.append(skaco_cost)


    ACA_runs = np.array(ACA_runs)
    mmACA_runs = np.array(mmACA_runs)
    PGACAfb_runs = np.array(PGACAfb_runs)
    PGACAb_runs = np.array(PGACAb_runs)
    # CPGACA_runs = np.array(CPGACA_runs)
    # GDACA_runs = np.array(GDACA_runs)

    print(f"    ACA: {ACA_runs.mean():.2f} +/- {ACA_runs.std():.2f}")
    print(f"  mmACA: {mmACA_runs.mean():.2f} +/- {mmACA_runs.std():.2f}")
    print(f"PGACAfb: {PGACAfb_runs.mean():.2f} +/- {PGACAfb_runs.std():.2f}")
    # print(f" PGACAf: { CPGACA_runs.mean():.2f} +/- { CPGACA_runs.std():.2f}")
    # print(f" PGACAf: {GDACA_runs.mean():.2f} +/- {GDACA_runs.std():.2f}")
    print(f" PGACAb: {PGACAb_runs.mean():.2f} +/- {PGACAb_runs.std():.2f}")

    from scipy import stats
    ttest, pval = stats.ttest_rel(ACA_runs, PGACAfb_runs)
    print("p-value", pval)
    # ttest, pval = stats.ttest_rel(ACA_runs, CPGACA_runs)
    # print("p-value", pval)
    ttest, pval = stats.ttest_rel(ACA_runs, PGACAb_runs)
    print("p-value", pval)


def plotting():
    """Function for quick plotting some results."""
    colors = {
        "ACA": ("blue", "blue"),
        "PGACA": ("green", "green"),
        "CPGACA": ("red", "red"),
        "GDACA": ("purple", "purple")
    }

    # problem space:
    size = 10
    runs = 10
    iterations = 1000
    distance_matrix = np.random.randint(0, 10, size**2).reshape((size, size))
    # distance_matrix[np.where(distance_matrix == 0)] = 1e13

    def cal_total_distance(routine):
        return sum([distance_matrix[routine[i % size], routine[(i + 1) % size]]
                    for i in range(size)])

    # PGACA
    print("Running ACA")
    aca_y = []
    for _ in tqdm(range(runs)):
        aca = ACA_TSP(func=cal_total_distance,
                      distance_matrix=distance_matrix,
                      params={
                          "max_iter": iterations,
                          "min_tau": 1e-13,
                          "learning_rate": 1
                      })
        aca.run(metric = ["branching_factor"])
        aca_y.append(aca.branching_factor)

    # PGACA
    print("Running PGACA")
    pgaca_y = []
    for _ in tqdm(range(runs)):
        aca = ACA_TSP(func=cal_total_distance,
                      distance_matrix=distance_matrix,
                      params={
                          "max_iter": iterations,
                          "min_tau": 1e-13,
                          "learning_rate": 1,
                          "advantage_type": "baseline"
                      })
        aca.run(metric = ["branching_factor"])
        pgaca_y.append(aca.branching_factor)

    # PGACA
    print("Running PGACA")
    pgaca_y1 = []
    for _ in tqdm(range(runs)):
        aca = ACA_TSP(func=cal_total_distance,
                      distance_matrix=distance_matrix,
                      params={
                          "max_iter": iterations,
                          "min_tau": 1e-13,
                          "learning_rate": 1,
                          "advantage_type": "forward_backward"
                      })
        aca.run(metric = ["branching_factor"])
        pgaca_y1.append(aca.branching_factor)

    # C-PGACA
    # print("Running C_PGACA")
    # cpgaca_y = []
    # for _ in tqdm(range(runs)):
    #     aca = ACA_TSP(func=cal_total_distance,
    #                   distance_matrix=distance_matrix,
    #                   params={
    #                       "max_iter": iterations,
    #                       "min_tau": 1e-13,
    #                       "learning_rate": 1
    #                   })
    #     aca.run()
    #     cpgaca_y.append(aca.current_best_Y)

    # C-PGACA
    # print("Running GDACA")
    # gdaca_y = []
    # for _ in tqdm(range(runs)):
    #     aca = GDACA(func=cal_total_distance,
    #                     distance_matrix=distance_matrix,
    #                     params={
    #                         "max_iter": iterations,
    #                         "min_tau": 1e-13,
    #                         "learning_rate": 1
    #                     })
    #     aca.run()
    #     gdaca_y.append(aca.current_best_Y)

    # plt.fill_between(np.arange(0, iterations), np.array(cpgaca_y).max(axis=0), np.array(cpgaca_y).min(axis=0), alpha=0.2, color=colors["CPGACA"][1], label="C-PGACA")
    # plt.fill_between(np.arange(0, iterations), np.array(pgaca_y).max(axis=0), np.array(pgaca_y).min(axis=0),   alpha=0.2, color=colors["PGACA"][1],  label="PGACA-fb")
    # plt.fill_between(np.arange(0, iterations), np.array(pgaca_y1).max(axis=0), np.array(pgaca_y1).min(axis=0),   alpha=0.2, color=colors["CPGACA"][1],  label="PGACA-baseline")
    # plt.fill_between(np.arange(0, iterations), np.array(aca_y).max(axis=0), np.array(aca_y).min(axis=0),       alpha=0.2, color=colors["ACA"][1],    label="ACA")
    # plt.fill_between(np.arange(0, iterations), np.array(gdaca_y).max(axis=0), np.array(gdaca_y).min(axis=0),   alpha=0.2, color=colors["GDACA"][1],    label="GDACA")


#     plt.plot(np.array(pgaca_y).max(axis=0), c=colors["PGACA"][0])
#     plt.plot(np.array(pgaca_y).min(axis=0), c=colors["PGACA"][0])
    plt.plot(np.median(pgaca_y, axis=0), c=colors["PGACA"][0], label="PGACA-fb")
    plt.plot(np.median(pgaca_y1, axis=0), c=colors["CPGACA"][0], label="PGACA-baseline")
    plt.plot(np.median(aca_y, axis=0), c=colors["ACA"][0], label="ACA")


#     plt.plot(np.array(pgaca_y1).max(axis=0), c=colors["CPGACA"][0])
#     plt.plot(np.array(pgaca_y1).min(axis=0), c=colors["CPGACA"][0])
#     plt.plot(np.array(pgaca_y1).mean(axis=0), c=colors["CPGACA"][0])

#    plt.plot(np.array(aca_y).max(axis=0), c=colors["ACA"][0])
#    plt.plot(np.array(aca_y).min(axis=0), c=colors["ACA"][0])
#     plt.plot(np.array(aca_y).mean(axis=0), c=colors["ACA"][0])

    # plt.plot(np.array(cpgaca_y).max(axis=0), c=colors["CPGACA"][0])
    # plt.plot(np.array(cpgaca_y).min(axis=0), c=colors["CPGACA"][0])

    # plt.plot(np.array(gdaca_y).max(axis=0), c=colors["GDACA"][0])
    # plt.plot(np.array(gdaca_y).min(axis=0), c=colors["GDACA"][0])

    plt.legend()
    plt.show()


if __name__ == "__main__":
    stat_run()
