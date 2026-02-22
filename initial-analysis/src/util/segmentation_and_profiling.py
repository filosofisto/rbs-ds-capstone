import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import seaborn as sns
from util.general_utilities import info, warning


# def find_optimal_k(company_df: pd.DataFrame, k_min=2, k_max=11):
#     """
#     Standalone function to compute elbow and silhouette for K-means.
#     Handles column names carefully after aggregation.
#     """
#     print("Columns in company_df before preprocessing:")
#     print(company_df.columns.tolist())
#
#     # Only rename lambda-style columns if they exist (safe)
#     rename_map = {}
#     for col in company_df.columns:
#         if '<lambda>' in col or col.startswith('season_') or col.startswith('quarter_'):
#             base = col.split('_')[0] if '_' in col else col
#             rename_map[col] = base
#
#     if rename_map:
#         company_df = company_df.rename(columns=rename_map)
#         print("\nRenamed lambda columns to:", rename_map)
#
#     print("\nColumns after safe rename (if any):")
#     print(company_df.columns.tolist())
#
#     # Define features using the expected names (adjust if your aggregation uses different)
#     num_features = ['total_purchase', 'avg_purchase', 'purchase_cv',
#                     'employees', 'revenue', 'active_quarters']  # or 'active_seasons'
#     cat_features = ['sector', 'province', 'legal_nature']
#
#     # Check for missing features
#     missing_num = [f for f in num_features if f not in company_df.columns]
#     missing_cat = [f for f in cat_features if f not in company_df.columns]
#     if missing_num or missing_cat:
#         print("\nMissing numerical features:", missing_num)
#         print("Missing categorical features:", missing_cat)
#         raise ValueError("Missing required columns — check aggregation step")
#
#     # Preprocessing pipeline
#     from sklearn.preprocessing import StandardScaler, OneHotEncoder
#     from sklearn.compose import ColumnTransformer
#
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), num_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
#         ]
#     )
#
#     X = preprocessor.fit_transform(company_df)
#
#     # Clustering loop
#     # from sklearn.cluster import KMeans
#     # from sklearn.metrics import silhouette_score
#     # import matplotlib.pyplot as plt
#
#     inertias = []
#     sil_scores = []
#     k_range = range(k_min, k_max + 1)
#
#     for k in k_range:
#         kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#         labels = kmeans.fit_predict(X)
#         inertias.append(kmeans.inertia_)
#         sil_scores.append(silhouette_score(X, labels))
#
#     # Plot
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#
#     ax1.plot(k_range, inertias, 'bo-')
#     ax1.set_title('Elbow Method')
#     ax1.set_xlabel('Number of clusters (k)')
#     ax1.set_ylabel('Inertia')
#     ax1.grid(True, alpha=0.3)
#
#     ax2.plot(k_range, sil_scores, 'ro-')
#     ax2.set_title('Silhouette Score')
#     ax2.set_xlabel('Number of clusters (k)')
#     ax2.set_ylabel('Silhouette Score')
#     ax2.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.show()
#
#     # Table
#     metrics = pd.DataFrame({
#         'k': k_range,
#         'inertia': [round(i, 2) for i in inertias],
#         'silhouette': [round(s, 4) for s in sil_scores]
#     })
#     print("Clustering Quality Metrics:")
#     print(metrics)
#
#     best_sil_k = metrics.loc[metrics['silhouette'].idxmax(), 'k']
#     print(f"\nRecommended k (highest silhouette): {best_sil_k} (score = {metrics['silhouette'].max():.4f})")
#
#     return metrics, best_sil_k

# def segmentation_and_profiling(df: pd.DataFrame):
#     """
#     Customer segmentation and profiling at company level.
#     """
#     # Aggregate to company level
#     company_df = df.groupby('company_id').agg(
#         total_purchase=('Purchase', 'sum'),
#         avg_purchase=('Purchase', 'mean'),
#         std_purchase=('Purchase', 'std'),
#         employees=('employees', 'mean'),
#         revenue=('revenue', 'mean'),
#         sector=('sector', 'first'),
#         province=('province', 'first'),
#         legal_nature=('legal_nature', 'first')
#     ).reset_index()
#
#     # Add CV (coefficient of variation of monthly purchases)
#     company_df['purchase_cv'] = (
#         company_df['std_purchase'] / company_df['avg_purchase'] * 100
#     ).fillna(0)
#
#     # Verify required columns
#     required_cols = ['total_purchase', 'avg_purchase', 'purchase_cv',
#                      'employees', 'revenue', 'sector', 'province', 'legal_nature']
#     missing = [col for col in required_cols if col not in company_df.columns]
#     if missing:
#         warning(f"Missing columns in company_df: {missing}")
#         warning(f"Available columns: {company_df.columns.tolist()}")
#         raise ValueError("Cannot proceed — missing required columns")
#
#     # Preprocessing pipeline
#     num_features = ['total_purchase', 'avg_purchase', 'purchase_cv',
#                     'employees', 'revenue']
#     cat_features = ['sector', 'province', 'legal_nature']
#
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), num_features),
#             ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
#         ]
#     )
#
#     X = preprocessor.fit_transform(company_df)
#
#     # Elbow + Silhouette to choose k
#     inertias = []
#     sil_scores = []
#     k_range = range(2, 11)
#
#     for k in k_range:
#         kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#         labels = kmeans.fit_predict(X)
#         inertias.append(kmeans.inertia_)
#         sil_scores.append(silhouette_score(X, labels))
#
#     # Plot elbow and silhouette
#     plt.close('all')
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
#
#     ax1.plot(k_range, inertias, 'bo-')
#     ax1.set_title('Elbow Method')
#     ax1.set_xlabel('Number of clusters (k)')
#     ax1.set_ylabel('Inertia')
#     ax1.grid(True, alpha=0.3)
#
#     ax2.plot(k_range, sil_scores, 'ro-')
#     ax2.set_title('Silhouette Score')
#     ax2.set_xlabel('Number of clusters (k)')
#     ax2.set_ylabel('Silhouette Score')
#     ax2.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.show()
#
#     # Print metrics table
#     metrics = pd.DataFrame({
#         'k': k_range,
#         'inertia': [round(i, 2) for i in inertias],
#         'silhouette': [round(s, 4) for s in sil_scores]
#     })
#     info("Clustering Quality Metrics:")
#     info(metrics)
#
#     # Choose optimal k (highest silhouette or elbow bend)
#     best_sil_k = metrics.loc[metrics['silhouette'].idxmax(), 'k']
#     info(f"\nRecommended k (highest silhouette): {best_sil_k} (score = {metrics['silhouette'].max():.4f})")
#
#     # Run clustering with chosen k
#     optimal_k = best_sil_k  # or set manually e.g. 4
#     kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
#     company_df['cluster'] = kmeans.fit_predict(X)
#
#     # Profile clusters
#     profile = company_df.groupby('cluster').agg({
#         'total_purchase': ['mean', 'sum'],
#         'avg_purchase': 'mean',
#         'purchase_cv': 'mean',
#         'employees': 'mean',
#         'revenue': 'mean',
#         'sector': lambda x: x.mode()[0] if not x.empty else 'N/A',
#         'province': lambda x: x.mode()[0] if not x.empty else 'N/A',
#         'legal_nature': lambda x: x.mode()[0] if not x.empty else 'N/A',
#         'company_id': 'count'
#     })
#
#     # Flatten columns
#     profile.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
#                        for col in profile.columns.values]
#
#     # Rename
#     profile = profile.rename(columns={
#         'total_purchase_mean': 'avg_total_purchase',
#         'total_purchase_sum': 'total_purchase_sum',
#         'avg_purchase_mean': 'avg_monthly_purchase',
#         'purchase_cv_mean': 'avg_cv',
#         'employees_mean': 'avg_employees',
#         'revenue_mean': 'avg_revenue',
#         'company_id_count': 'n_companies'
#     })
#
#     # Revenue share
#     profile['revenue_share_%'] = (
#         profile['total_purchase_sum'] / profile['total_purchase_sum'].sum() * 100
#     ).round(1)
#
#     info("Cluster Profiles:")
#     print(profile.round(2).to_string())
#
#     # Plot: split into high-scale and low-scale to handle outliers
#     high_cols = ['avg_total_purchase', 'avg_monthly_purchase', 'total_purchase_sum', 'avg_revenue']
#     low_cols = ['avg_cv', 'avg_employees', 'revenue_share_%']
#
#     # High-scale chart (log scale)
#     plt.close('all')
#     plt.figure(figsize=(12, 7))
#     profile[high_cols].plot(kind='bar', cmap='Blues', logy=True)
#     plt.title('High-Scale Metrics per Cluster (Log Scale)')
#     plt.ylabel('Value (€) – Log Scale')
#     plt.xlabel('Cluster')
#     plt.xticks(rotation=0)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()
#
#     # Low-scale chart (linear)
#     plt.close('all')
#     plt.figure(figsize=(12, 7))
#     profile[low_cols].plot(kind='bar', cmap='Greens')
#     plt.title('Other Metrics per Cluster')
#     plt.ylabel('Value')
#     plt.xlabel('Cluster')
#     plt.xticks(rotation=0)
#     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.show()
#
#     return company_df

def segmentation_and_profiling(df: pd.DataFrame):
    """
    Customer segmentation and profiling at company level.
    Uses explicit figure/axes to avoid empty phantom figures.
    """
    # Aggregate to company level
    company_df = df.groupby('company_id').agg(
        total_purchase=('Purchase', 'sum'),
        avg_purchase=('Purchase', 'mean'),
        std_purchase=('Purchase', 'std'),
        employees=('employees', 'mean'),
        revenue=('revenue', 'mean'),
        sector=('sector', 'first'),
        province=('province', 'first'),
        legal_nature=('legal_nature', 'first')
    ).reset_index()

    # Add CV (coefficient of variation of monthly purchases)
    company_df['purchase_cv'] = (
        company_df['std_purchase'] / company_df['avg_purchase'] * 100
    ).fillna(0)

    # Verify required columns
    required_cols = ['total_purchase', 'avg_purchase', 'purchase_cv',
                     'employees', 'revenue', 'sector', 'province', 'legal_nature']
    missing = [col for col in required_cols if col not in company_df.columns]
    if missing:
        print(f"Missing columns in company_df: {missing}")
        print(f"Available columns: {company_df.columns.tolist()}")
        raise ValueError("Cannot proceed — missing required columns")

    # Preprocessing pipeline
    num_features = ['total_purchase', 'avg_purchase', 'purchase_cv',
                    'employees', 'revenue']
    cat_features = ['sector', 'province', 'legal_nature']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    X = preprocessor.fit_transform(company_df)

    # Elbow + Silhouette to choose k
    inertias = []
    sil_scores = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(X, labels))

    # Plot elbow and silhouette (explicit figure)
    plt.close('all')
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(k_range, inertias, 'bo-')
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.grid(True, alpha=0.3)

    ax2.plot(k_range, sil_scores, 'ro-')
    ax2.set_title('Silhouette Score')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    plt.close('all')

    # Print metrics table
    metrics = pd.DataFrame({
        'k': k_range,
        'inertia': [round(i, 2) for i in inertias],
        'silhouette': [round(s, 4) for s in sil_scores]
    })
    print("Clustering Quality Metrics:")
    print(metrics)

    # Choose optimal k (highest silhouette)
    best_sil_k = metrics.loc[metrics['silhouette'].idxmax(), 'k']
    print(f"\nRecommended k (highest silhouette): {best_sil_k} (score = {metrics['silhouette'].max():.4f})")

    # Run clustering with chosen k
    optimal_k = best_sil_k  # or manually override, e.g. optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    company_df['cluster'] = kmeans.fit_predict(X)

    # Profile clusters
    profile = company_df.groupby('cluster').agg({
        'total_purchase': ['mean', 'sum'],
        'avg_purchase': 'mean',
        'purchase_cv': 'mean',
        'employees': 'mean',
        'revenue': 'mean',
        'sector': lambda x: x.mode()[0] if not x.empty else 'N/A',
        'province': lambda x: x.mode()[0] if not x.empty else 'N/A',
        'legal_nature': lambda x: x.mode()[0] if not x.empty else 'N/A',
        'company_id': 'count'
    })

    # Flatten columns
    profile.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col
                       for col in profile.columns.values]

    # Rename
    profile = profile.rename(columns={
        'total_purchase_mean': 'avg_total_purchase',
        'total_purchase_sum': 'total_purchase_sum',
        'avg_purchase_mean': 'avg_monthly_purchase',
        'purchase_cv_mean': 'avg_cv',
        'employees_mean': 'avg_employees',
        'revenue_mean': 'avg_revenue',
        'company_id_count': 'n_companies'
    })

    # Revenue share
    profile['revenue_share_%'] = (
        profile['total_purchase_sum'] / profile['total_purchase_sum'].sum() * 100
    ).round(1)

    print("\nCluster Profiles:")
    print(profile.round(2).to_string())

    # ──────────────────────────────────────────────────────────────
    # PLOTS: explicit figures + close after show
    # ──────────────────────────────────────────────────────────────

    high_cols = ['avg_total_purchase', 'avg_monthly_purchase', 'total_purchase_sum', 'avg_revenue']
    low_cols = ['avg_cv', 'avg_employees', 'revenue_share_%']

    # High-scale chart (log scale)
    plt.close('all')
    fig_high = plt.figure(figsize=(12, 7))
    ax_high = fig_high.add_subplot(111)
    profile[high_cols].plot(kind='bar', cmap='Blues', logy=True, ax=ax_high)
    ax_high.set_title('High-Scale Metrics per Cluster (Log Scale)')
    ax_high.set_ylabel('Value (€) – Log Scale')
    ax_high.set_xlabel('Cluster')
    ax_high.tick_params(axis='x', rotation=0)
    ax_high.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_high.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close('all')

    # Low-scale chart (linear)
    plt.close('all')
    fig_low = plt.figure(figsize=(12, 7))
    ax_low = fig_low.add_subplot(111)
    profile[low_cols].plot(kind='bar', cmap='Greens', ax=ax_low)
    ax_low.set_title('Other Metrics per Cluster')
    ax_low.set_ylabel('Value')
    ax_low.set_xlabel('Cluster')
    ax_low.tick_params(axis='x', rotation=0)
    ax_low.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_low.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.close('all')

    return company_df


def visualize_clusters_with_pca(company_df: pd.DataFrame, n_components=2):
    """
    Visualizes clusters using PCA reduced to 2D, and shows feature loadings.

    Parameters:
    - company_df: DataFrame with 'cluster' column already assigned
    - n_components: Number of principal components (usually 2 for visualization)
    """
    # Features used in clustering
    num_features = ['total_purchase', 'avg_purchase', 'purchase_cv',
                    'employees', 'revenue']
    cat_features = ['sector', 'province', 'legal_nature']

    # Preprocessor (must match exactly what was used for clustering)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    # Transform data
    X = preprocessor.fit_transform(company_df)

    # Apply PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    # Explained variance
    explained_variance = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA explained variance ({n_components} components): {explained_variance:.1f}%")

    # Create DataFrame for scatter plot
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['cluster'] = company_df['cluster'].astype(str)

    # Scatter plot of clusters
    plt.figure(figsize=(12, 9))
    sns.scatterplot(
        data=pca_df,
        x='PC1',
        y='PC2',
        hue='cluster',
        palette='tab10',
        s=80,
        alpha=0.8,
        edgecolor='black'
    )
    plt.title(f'PCA Visualization of Clusters (Explained Variance: {explained_variance:.1f}%)', fontsize=14)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # === LOADINGS ANALYSIS ===
    # Get feature names after preprocessing (includes encoded categories)
    feature_names = preprocessor.get_feature_names_out()

    # Loadings DataFrame
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=feature_names
    )

    print("\nPCA Loadings (Feature Contributions to Each Principal Component):")
    print("Sorted by absolute contribution to PC1:")
    print(loadings.abs().sort_values('PC1', ascending=False).round(3).head(15))

    # Bar plots of top contributors
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharey=False)

    # PC1 top features
    loadings['PC1'].abs().sort_values(ascending=False).head(10).plot(
        kind='barh', ax=axes[0], color='skyblue'
    )
    axes[0].set_title('Top 10 Features Contributing to PC1', fontsize=13)
    axes[0].set_xlabel('Absolute Loading', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='x')

    # PC2 top features
    loadings['PC2'].abs().sort_values(ascending=False).head(10).plot(
        kind='barh', ax=axes[1], color='lightgreen'
    )
    axes[1].set_title('Top 10 Features Contributing to PC2', fontsize=13)
    axes[1].set_xlabel('Absolute Loading', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.suptitle('Feature Importance in Principal Components', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def segmentation_and_profiling_with_season(df: pd.DataFrame):
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
