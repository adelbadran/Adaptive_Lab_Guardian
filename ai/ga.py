import numpy as np
from rbf import RBFNetwork
from preprocessing import preprocess_data


class GA:
    def __init__(self, pop_size=10, generations=10):
        self.pop_size = pop_size
        self.generations = generations

    # 🔹 إنشاء population عشوائي
    def init_population(self):
        population = []
        for _ in range(self.pop_size):
            centers = np.random.randint(5, 20)
            sigma = np.random.uniform(0.5, 3)
            population.append([centers, sigma])
        return population

    # 🔹 تقييم الفرد
    def fitness(self, individual, X_train, y_train, X_test, y_test):
        centers, sigma = int(individual[0]), individual[1]

        model = RBFNetwork(num_centers=centers, sigma=sigma)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        preds_classes = (preds > 0.5).astype(int)

        acc = np.mean(preds_classes == y_test)
        return acc

    # 🔹 اختيار الأفضل
    def selection(self, population, scores):
        sorted_idx = np.argsort(scores)[::-1]
        return [population[i] for i in sorted_idx[:4]]

    # 🔹 crossover
    def crossover(self, p1, p2):
        return [
            p1[0] if np.random.rand() < 0.5 else p2[0],
            p1[1] if np.random.rand() < 0.5 else p2[1]
        ]

    # 🔹 mutation
    def mutate(self, individual):
        if np.random.rand() < 0.6:
            individual[0] = np.random.randint(5, 20)
        if np.random.rand() < 0.6:
            individual[1] = np.random.uniform(0.5, 3)
        return individual

    # 🔹 التشغيل الأساسي
    def run(self, X_train, y_train, X_test, y_test):
        population = self.init_population()

        best = None
        best_score = 0

        for gen in range(self.generations):
            scores = []

            print(f"\n🔄 Generation {gen}")

            for ind in population:
                score = self.fitness(ind, X_train, y_train, X_test, y_test)
                scores.append(score)
                print(f"Individual {ind} → Acc = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best = ind

            print(f"⭐ Best so far: {best} → {best_score:.4f}")

            # 🔹 selection
            parents = self.selection(population, scores)

            # 🔹 new population
            new_population = parents.copy()

            while len(new_population) < self.pop_size:

                # 30% random individuals
                if np.random.rand() < 0.3:
                    new_population.append([
                        np.random.randint(5, 20),
                        np.random.uniform(0.5, 3)
                    ])
                else:
                    p1 = parents[np.random.randint(len(parents))]
                    p2 = parents[np.random.randint(len(parents))]

                    child = self.crossover(p1, p2)
                    child = self.mutate(child)
                    new_population.append(child)

            population = new_population

        return best, best_score


# =====================================
# 🚀 MAIN
# =====================================

if __name__ == "__main__":

    print("🔥 Running Genetic Algorithm...\n")

    X_train, X_test, y_train, y_test, *_ = preprocess_data(
        csv_path="data/Adaptive_Lab_Guardian.csv"
    )

    ga = GA(pop_size=10, generations=10)

    best, score = ga.run(X_train, y_train, X_test, y_test)

    print("\n🎯 FINAL RESULT")
    print("Best Centers:", best[0])
    print("Best Sigma:", best[1])
    print("Best Accuracy:", score)