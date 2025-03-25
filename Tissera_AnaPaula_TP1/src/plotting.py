import matplotlib.pyplot as plt
import seaborn as sns

def plot_distributions(df, dset):
    to_plot = df.select_dtypes(include='number')

    plt.figure(figsize=(12, 10))
    for i, col in enumerate(to_plot, 1):
        plt.subplot(3, 4, i)
        sns.histplot(df[col], kde=True, color="steelblue")
        plt.title(col)
    plt.tight_layout()
    plt.suptitle(f"Distribution of numerical variables for the {dset} Set", y=1.02, size=15)
    plt.show()
    
def plot_scatter_matrix(df, dset):
    to_plot = df.select_dtypes(include='number')
    g = sns.pairplot(to_plot)
    g.fig.suptitle(f"Scatterplot Matrix for the {dset} Set", y=1.02, size=25) 
    plt.show()
    
def plot_correlation(df, dset):
    df_numeric = df.select_dtypes(include=['number'])
    correlation_matrix = df_numeric.corr()

    plt.figure(figsize=(10, 8)) 
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title(f"Correlation Heatmap for {dset}", size=15)
    plt.show()
    
def plot_rooms_vs_area(df):
    plt.figure(figsize=(5,4))
    plt.scatter(df["area"], df["rooms"], alpha=0.6, edgecolors="k", color = "cadetblue")

    plt.xlabel("Área (m²)")
    plt.ylabel("Número de habitaciones")
    plt.title(f"Distribución de Rooms vs Área")
    plt.grid(True)

    plt.show()
    
def plot_age_vs(df, features):
    fig, axes = plt.subplots(1, len(features), figsize=(15, 4))

    for i, feature in enumerate(features):
        axes[i].scatter(df[feature], df["age"], alpha=0.5, color= "teal")
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel("age")
        axes[i].set_title(f"age vs {feature}")

    plt.tight_layout()
    plt.show()
    
def coeficient_evolution(coefficients, lambdas, l_type, m_type):
    
    plt.figure(figsize=(10, 6))
    for i in range(coefficients.shape[1]):
        plt.plot(lambdas, coefficients[:, i], label=f'w{i}')

    plt.xscale('log')
    plt.xlabel(f"Lambda (regularization) {l_type}")
    plt.ylabel('Value of the coefficients w*')
    plt.title(f"Evolution of the coefficients as a function of λ with a {m_type} model")
    plt.legend()
    plt.show()
    