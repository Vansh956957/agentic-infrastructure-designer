import { Node, Connection, Group } from './types';
import { NODE_WIDTH, NODE_HEIGHT } from './constants';

const PIPELINE_CODE = `
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load Dataset
df = pd.read_csv('dataset.csv')
print("Dataset Loaded:", df.shape)

# 2. Features & Target
X = df.drop('target', axis=1)
y = df['target']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
`;

const getModelCode = (modelImport: string, modelInit: string, isRegression: boolean) => {
    const metric = isRegression ? 'mean_squared_error, r2_score' : 'accuracy_score, classification_report';
    const evalPrint = isRegression 
        ? `print(f"MSE: {mean_squared_error(y_test, y_pred)}")\nprint(f"R2: {r2_score(y_test, y_pred)}")` 
        : `print(f"Accuracy: {accuracy_score(y_test, y_pred)}")\nprint(classification_report(y_test, y_pred))`;

    return `${PIPELINE_CODE}

# 5. Model Initialization
${modelImport}
model = ${modelInit}

# 6. Training
model.fit(X_train_scaled, y_train)

# 7. Prediction
y_pred = model.predict(X_test_scaled)

# 8. Evaluation
from sklearn.metrics import ${metric}
${evalPrint}
`;
};

const COLORS = {
    pipeline: 'bg-blue-100 border-blue-500 shadow-blue-500/20',
    regression: 'bg-green-100 border-green-500 shadow-green-500/20',
    classification: 'bg-orange-100 border-orange-500 shadow-orange-500/20',
    unsupervised: 'bg-purple-100 border-purple-500 shadow-purple-500/20',
    rl: 'bg-yellow-100 border-yellow-500 shadow-yellow-500/20',
    root: 'bg-gray-800 border-gray-600 text-white',
};

export const getAIState = () => {
    let nodes: Node[] = [];
    let connections: Connection[] = [];
    let groups: Group[] = [];
    let nId = 1;
    let cId = 1;
    let gId = 1;

    const createNode = (label: string, type: string, x: number, y: number, theme: string, code: string = ''): Node => ({
        id: `ai-node-${nId++}`,
        type: 'text', // Using text type generic styling for now, but overriding with customTheme
        position: { x, y },
        data: { label, customTheme: `${theme} border-2 rounded-lg font-bold`, code },
        isAnimating: false,
    });

    const createConn = (from: string, to: string) => {
        connections.push({
            id: `ai-conn-${cId++}`,
            fromNodeId: from,
            toNodeId: to,
            isAnimating: false
        });
    };

    // --- ROOT ---
    const rootNode = createNode("Machine Learning", 'text', 800, 50, COLORS.root);
    nodes.push(rootNode);

    // --- COMPARTMENTS (Groups acting as headers for sections) ---
    // Supervised
    const supervisedNode = createNode("Supervised Learning", 'text', 400, 200, 'bg-blue-50 border-blue-300');
    createConn(rootNode.id, supervisedNode.id);
    nodes.push(supervisedNode);

    // Unsupervised
    const unsupervisedNode = createNode("Unsupervised Learning", 'text', 1100, 200, 'bg-purple-50 border-purple-300');
    createConn(rootNode.id, unsupervisedNode.id);
    nodes.push(unsupervisedNode);

    // RL
    const rlNode = createNode("Reinforcement Learning", 'text', 1500, 200, 'bg-yellow-50 border-yellow-300');
    createConn(rootNode.id, rlNode.id);
    nodes.push(rlNode);

    // --- SUPERVISED PIPELINE ---
    const pipeX = 400;
    let pipeY = 350;
    const pipelineSteps = [
        { label: "1. Imports & Config", code: "import pandas as pd\nimport numpy as np\n..." },
        { label: "2. Read Dataset", code: "df = pd.read_csv('data.csv')" },
        { label: "3. Feature Split (X/y)", code: "X = df.drop('target', axis=1)\ny = df['target']" },
        { label: "4. Train/Test Split", code: "X_train, X_test, y_train, y_test = train_test_split(X, y)" },
    ];

    let lastPipeId = supervisedNode.id;
    pipelineSteps.forEach(step => {
        const node = createNode(step.label, 'text', pipeX, pipeY, COLORS.pipeline, PIPELINE_CODE);
        nodes.push(node);
        createConn(lastPipeId, node.id);
        lastPipeId = node.id;
        pipeY += 120;
    });

    // --- REGRESSION BRANCH ---
    const regX = 100;
    let regY = pipeY + 100;
    const regHeader = createNode("Regression Models", 'text', regX + 100, pipeY, COLORS.regression);
    nodes.push(regHeader);
    createConn(lastPipeId, regHeader.id);

    const regressionModels = [
        { name: "Linear Regression", import: "from sklearn.linear_model import LinearRegression", init: "LinearRegression()" },
        { name: "Polynomial Regression", import: "from sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.pipeline import make_pipeline", init: "make_pipeline(PolynomialFeatures(degree=2), LinearRegression())" },
        { name: "Ridge Regression", import: "from sklearn.linear_model import Ridge", init: "Ridge(alpha=1.0)" },
        { name: "Lasso Regression", import: "from sklearn.linear_model import Lasso", init: "Lasso(alpha=0.1)" },
        { name: "ElasticNet", import: "from sklearn.linear_model import ElasticNet", init: "ElasticNet(alpha=0.1, l1_ratio=0.5)" },
        { name: "SVR", import: "from sklearn.svm import SVR", init: "SVR(kernel='rbf')" },
        { name: "Decision Tree Regressor", import: "from sklearn.tree import DecisionTreeRegressor", init: "DecisionTreeRegressor(max_depth=5)" },
        { name: "Random Forest Regressor", import: "from sklearn.ensemble import RandomForestRegressor", init: "RandomForestRegressor(n_estimators=100)" },
        { name: "Gradient Boosting Regressor", import: "from sklearn.ensemble import GradientBoostingRegressor", init: "GradientBoostingRegressor(n_estimators=100)" },
        { name: "XGBoost Regressor", import: "from xgboost import XGBRegressor", init: "XGBRegressor(n_estimators=100)" },
        { name: "LightGBM Regressor", import: "from lightgbm import LGBMRegressor", init: "LGBMRegressor()" },
        { name: "CatBoost Regressor", import: "from catboost import CatBoostRegressor", init: "CatBoostRegressor(verbose=0)" },
        { name: "KNN Regressor", import: "from sklearn.neighbors import KNeighborsRegressor", init: "KNeighborsRegressor(n_neighbors=5)" },
    ];

    regressionModels.forEach((m, idx) => {
        // Stagger positions slightly for tree effect
        const xPos = regX + (idx % 2 === 0 ? 0 : 250);
        const yPos = regY + (Math.floor(idx / 2) * 100);
        
        const node = createNode(m.name, 'text', xPos, yPos, COLORS.regression, getModelCode(m.import, m.init, true));
        nodes.push(node);
        createConn(regHeader.id, node.id);
    });


    // --- CLASSIFICATION BRANCH ---
    const clfX = 700;
    let clfY = pipeY + 100;
    const clfHeader = createNode("Classification Models", 'text', clfX + 100, pipeY, COLORS.classification);
    nodes.push(clfHeader);
    createConn(lastPipeId, clfHeader.id);

    const classificationModels = [
        { name: "Logistic Regression", import: "from sklearn.linear_model import LogisticRegression", init: "LogisticRegression()" },
        { name: "KNN Classifier", import: "from sklearn.neighbors import KNeighborsClassifier", init: "KNeighborsClassifier(n_neighbors=5)" },
        { name: "SVM Classifier", import: "from sklearn.svm import SVC", init: "SVC(kernel='rbf', probability=True)" },
        { name: "Naive Bayes (Gaussian)", import: "from sklearn.naive_bayes import GaussianNB", init: "GaussianNB()" },
        { name: "Naive Bayes (Bernoulli)", import: "from sklearn.naive_bayes import BernoulliNB", init: "BernoulliNB()" },
        { name: "Naive Bayes (Multinomial)", import: "from sklearn.naive_bayes import MultinomialNB", init: "MultinomialNB()" },
        { name: "Decision Tree Classifier", import: "from sklearn.tree import DecisionTreeClassifier", init: "DecisionTreeClassifier(max_depth=5)" },
        { name: "Random Forest Classifier", import: "from sklearn.ensemble import RandomForestClassifier", init: "RandomForestClassifier(n_estimators=100)" },
        { name: "Gradient Boosting Classifier", import: "from sklearn.ensemble import GradientBoostingClassifier", init: "GradientBoostingClassifier()" },
        { name: "XGBoost Classifier", import: "from xgboost import XGBClassifier", init: "XGBClassifier()" },
        { name: "LightGBM Classifier", import: "from lightgbm import LGBMClassifier", init: "LGBMClassifier()" },
        { name: "CatBoost Classifier", import: "from catboost import CatBoostClassifier", init: "CatBoostClassifier(verbose=0)" },
        { name: "MLP Classifier", import: "from sklearn.neural_network import MLPClassifier", init: "MLPClassifier(hidden_layer_sizes=(100,50))" },
        { name: "LDA", import: "from sklearn.discriminant_analysis import LinearDiscriminantAnalysis", init: "LinearDiscriminantAnalysis()" },
        { name: "QDA", import: "from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis", init: "QuadraticDiscriminantAnalysis()" },
    ];

    classificationModels.forEach((m, idx) => {
         // Stagger positions
        const xPos = clfX + (idx % 2 === 0 ? 0 : 250);
        const yPos = clfY + (Math.floor(idx / 2) * 100);

        const node = createNode(m.name, 'text', xPos, yPos, COLORS.classification, getModelCode(m.import, m.init, false));
        nodes.push(node);
        createConn(clfHeader.id, node.id);
    });

    // --- UNSUPERVISED LIST ---
    const unsupAlgos = ["KMeans", "DBSCAN", "Hierarchical Clustering", "PCA", "t-SNE", "UMAP", "Gaussian Mixture Models"];
    let unsupY = 350;
    unsupAlgos.forEach(name => {
        const node = createNode(name, 'text', 1100, unsupY, COLORS.unsupervised, "# Code template coming soon...");
        nodes.push(node);
        createConn(unsupervisedNode.id, node.id);
        unsupY += 100;
    });

    // --- RL LIST ---
    const rlAlgos = ["Q-Learning", "Deep Q Networks (DQN)", "SARSA", "PPO", "A3C", "Monte Carlo RL"];
    let rlY = 350;
    rlAlgos.forEach(name => {
        const node = createNode(name, 'text', 1500, rlY, COLORS.rl, "# Code template coming soon...");
        nodes.push(node);
        createConn(rlNode.id, node.id);
        rlY += 100;
    });

    return { nodes, connections, groups };
};
