import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from util.general_utilities import info


def segmentation_and_profiling(df: pd.DataFrame):
    # Aggregate to company level — use named aggregation to avoid multi-index issues
    company_df = df.groupby('company_id').agg(
        total_purchase=('Purchase', 'sum'),
        avg_purchase=('Purchase', 'mean'),
        std_purchase=('Purchase', 'std'),
        employees=('employees', 'mean'),
        revenue=('revenue', 'mean'),
        sector=('sector', 'first'),
        province=('province', 'first'),
        legal_nature=('legal_nature', 'first'),
        active_seasons=('season', 'nunique')  # number of distinct seasons
    ).reset_index()

    # Add CV (coefficient of variation)
    company_df['purchase_cv'] = (
        company_df['std_purchase'] / company_df['avg_purchase'] * 100
    ).fillna(0)

    # Verify columns exist before preprocessing
    required_cols = ['total_purchase', 'avg_purchase', 'purchase_cv', 'employees', 'revenue',
                     'active_seasons', 'sector', 'province', 'legal_nature']
    missing = [col for col in required_cols if col not in company_df.columns]
    if missing:
        print(f"Missing columns in company_df: {missing}")
        print("Available columns:", company_df.columns.tolist())
        raise ValueError("Cannot proceed — missing required columns")

    # Preprocessing pipeline
    num_features = ['total_purchase', 'avg_purchase', 'purchase_cv', 'employees', 'revenue', 'active_seasons']
    cat_features = ['sector', 'province', 'legal_nature']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    X = preprocessor.fit_transform(company_df)

    # Clustering (use elbow/silhouette to choose k)
    inertias = []
    sil_scores = []
    k_range = range(2, 11)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X, kmeans.labels_))

    # Plot elbow and silhouette
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'bo-')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, sil_scores, 'ro-')
    plt.title('Silhouette Scores')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.tight_layout()
    plt.show()

    # Choose k (example: 4)
    optimal_k = 4  # ← update based on your plots
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    company_df['cluster'] = kmeans.fit_predict(X)

    # Profile clusters
    profile = company_df.groupby('cluster').agg({
        'total_purchase': ['mean', 'sum'],
        'avg_purchase': 'mean',
        'purchase_cv': 'mean',
        'employees': 'mean',
        'revenue': 'mean',
        'active_seasons': 'mean',
        'sector': lambda x: x.mode()[0] if not x.empty else 'N/A',
        'province': lambda x: x.mode()[0] if not x.empty else 'N/A',
        'legal_nature': lambda x: x.mode()[0] if not x.empty else 'N/A',
        'company_id': 'count'
    })

    # Flatten columns
    profile.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in profile.columns.values]

    # Rename
    profile = profile.rename(columns={
        'total_purchase_mean': 'avg_total_purchase',
        'total_purchase_sum': 'total_purchase_sum',
        'avg_purchase_mean': 'avg_monthly_purchase',
        'purchase_cv_mean': 'avg_cv',
        'employees_mean': 'avg_employees',
        'revenue_mean': 'avg_revenue',
        'active_seasons_mean': 'avg_active_seasons',
        'company_id_count': 'n_companies'
    })

    # Revenue share
    profile['revenue_share_%'] = (
        profile['total_purchase_sum'] / profile['total_purchase_sum'].sum() * 100
    ).round(1)

    info("Cluster Profiles:")
    print(profile.round(2).to_string())

    # Plot
    plot_cols = [
        'avg_total_purchase', 'avg_monthly_purchase', 'avg_cv',
        'avg_employees', 'avg_revenue', 'avg_active_seasons', 'revenue_share_%'
    ]
    profile[plot_cols].plot(kind='bar', figsize=(14, 8), cmap='tab20')
    plt.title('Key Metrics per Cluster')
    plt.ylabel('Value')
    plt.xlabel('Cluster')
    plt.xticks(rotation=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return company_df


def avg_purchase_by_cluster_and_season(original_df: pd.DataFrame, clustered_df: pd.DataFrame):
    for cluster in clustered_df['cluster'].unique():
        cluster_ids = clustered_df[clustered_df['cluster'] == cluster]['company_id']
        cluster_monthly = original_df[original_df['company_id'].isin(cluster_ids)]

        if cluster_monthly.empty:
            print(f"Cluster {cluster}: No data available")
            continue

        season_avg = cluster_monthly.groupby('season')['Purchase'].mean()
        print(f"Cluster {cluster} - Avg Purchase by Season:")
        print(season_avg.round(0))
        print("-" * 40)
