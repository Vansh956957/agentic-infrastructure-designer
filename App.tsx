
import React, { useState, useCallback, useRef, useEffect } from 'react';
import { LeftSidebar } from './components/LeftSidebar';
import { RightSidebar } from './components/RightSidebar';
import { Canvas } from './components/Canvas';
import { Node, Connection, Vector2D, AWSService, NodeData, Group, GroupData, Shape, ShapeData, TextNode, TextNodeData, Snapshot, Template, ValidationIssue, TrafficLevel, AWSCategory, WorkflowStatus } from './types';
import { ALL_SERVICES, NODE_WIDTH, NODE_HEIGHT, WORKFLOW_NODE_SIZE, ARCHITECTURE_TEMPLATES, AI_TEMPLATES } from './constants';
import { MenuIcon, EraseIcon, UndoIcon, RedoIcon, ZoomInIcon, ZoomOutIcon, CursorIcon, AlignLeftIcon, AlignCenterIcon, AlignRightIcon, AlignTopIcon, AlignMiddleIcon, AlignBottomIcon, DistributeHorizontalIcon, DistributeVerticalIcon, DoubleArrowIcon } from './components/Icons';
import { CommandPalette } from './components/CommandPalette';
import { validateGraph } from './validation';

declare var GIF: any;
declare var html2canvas: any;

type Selection = { type: 'node', id: string } | { type: 'connection', id: string } | { type: 'group', id: string } | { type: 'shape', id: string } | { type: 'text', id: string };

export type AppState = {
    nodes: Node[];
    connections: Connection[];
    groups: Group[];
    shapes: Shape[];
    textNodes: TextNode[];
    nextNodeId: number;
    nextConnectionId: number;
    nextGroupId: number;
    nextShapeId: number;
    nextTextNodeId: number;
}
type Toast = { id: number, message: string, color: string, duration: number };
type WorkflowStep = { type: 'node' | 'connection', id: string };

const HISTORY_LIMIT = 50;

const App: React.FC = () => {
  // Mode State: 'aws' | 'ai' | 'dl'
  const [appMode, setAppMode] = useState<'aws' | 'ai' | 'dl'>('aws');
  
  const [nodes, setNodes] = useState<Node[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [groups, setGroups] = useState<Group[]>([]);
  const [shapes, setShapes] = useState<Shape[]>([]);
  const [textNodes, setTextNodes] = useState<TextNode[]>([]);
  const [selection, setSelection] = useState<Selection[]>([]);
  const [isSelectingForExport, setIsSelectingForExport] = useState(false);
  
  // Independent Sidebar Control
  const [isLeftSidebarOpen, setIsLeftSidebarOpen] = useState(true);
  const [isRightSidebarOpen, setIsRightSidebarOpen] = useState(true);
  
  // Multi-Select Mode
  const [isMultiSelectMode, setIsMultiSelectMode] = useState(false);

  const [compareMode, setCompareMode] = useState(false); 
  const [compareNodeId, setCompareNodeId] = useState<string | null>(null); 

  // Tutorial State
  const [tutorialState, setTutorialState] = useState<{
      active: boolean;
      step: number;
      highlightNodeId: string | null;
      glowButton: 'compare' | 'tensorflow' | null;
      message: string | null;
  }>({ active: false, step: 0, highlightNodeId: null, glowButton: null, message: null });

  // Workflow / Animation State
  const [workflowStatus, setWorkflowStatus] = useState<WorkflowStatus>('idle');
  const [workflowQueue, setWorkflowQueue] = useState<WorkflowStep[]>([]);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [workflowRunningNodeId, setWorkflowRunningNodeId] = useState<string | null>(null);
  const workflowTimeoutRef = useRef<number | null>(null);
  const workflowCodeHistory = useRef<string[]>([]); // Tracks nodes with code during workflow
  const [cameraTransitionDuration, setCameraTransitionDuration] = useState(0);

  const [animationState, setAnimationState] = useState<{activeNodes: Set<string>, activeConnections: Set<string>, alertConnections: Set<string>}>({ activeNodes: new Set(), activeConnections: new Set(), alertConnections: new Set() });
  const [snapshots, setSnapshots] = useState<Snapshot[]>([]);
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [undoStack, setUndoStack] = useState<AppState[]>([]);
  const [redoStack, setRedoStack] = useState<AppState[]>([]);
  
  const addToast = useCallback((message: string, color: string = 'bg-green-500', duration: number = 4000) => {
      const id = Date.now() + Math.random();
      setToasts(prev => [...prev, { id, message, color, duration }]);
      setTimeout(() => {
          setToasts(current => current.filter(t => t.id !== id));
      }, duration);
  }, []);

  // Stored state for switching modes
  const [awsState, setAwsState] = useState<AppState | null>(null);
  const [aiState, setAiState] = useState<AppState | null>(null);
  const [dlState, setDlState] = useState<AppState | null>(null);

  // Validation State
  const [validationIssues, setValidationIssues] = useState<ValidationIssue[]>([]);
  const [selectedIssue, setSelectedIssue] = useState<ValidationIssue | null>(null);
  
  // Theme & View State
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [transform, setTransform] = useState({ x: 0, y: 0, k: 1 });
  
  // GIF Recording State
  const [isRecordingGif, setIsRecordingGif] = useState(false);

  const nextNodeId = useRef(0);
  const nextConnectionId = useRef(0);
  const nextGroupId = useRef(0);
  const nextShapeId = useRef(0);
  const nextTextNodeId = useRef(0);

  // --- Calculations for Floating Alignment Panel ---
  const selectedNodes = selection
        .filter(s => s.type === 'node')
        .map(s => nodes.find(n => n.id === s.id))
        .filter((n): n is Node => n !== undefined);

  const getSelectionBounds = () => {
      if (selectedNodes.length < 2) return null;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      
      selectedNodes.forEach(n => {
          const isWf = n.type === 'start' || n.type === 'end';
          const w = isWf ? WORKFLOW_NODE_SIZE : NODE_WIDTH;
          const h = isWf ? WORKFLOW_NODE_SIZE : NODE_HEIGHT;
          minX = Math.min(minX, n.position.x);
          minY = Math.min(minY, n.position.y);
          maxX = Math.max(maxX, n.position.x + w);
          maxY = Math.max(maxY, n.position.y + h);
      });
      return { x: minX, y: minY, w: maxX - minX, h: maxY - minY, maxX, minY };
  };

  const selectionBounds = getSelectionBounds();

  const getCurrentState = useCallback((): AppState => ({
      nodes, connections, groups, shapes, textNodes,
      nextNodeId: nextNodeId.current,
      nextConnectionId: nextConnectionId.current,
      nextGroupId: nextGroupId.current,
      nextShapeId: nextShapeId.current,
      nextTextNodeId: nextTextNodeId.current,
  }), [nodes, connections, groups, shapes, textNodes]);

  const loadState = useCallback((state: AppState) => {
      setNodes(state.nodes || []);
      setConnections(state.connections || []);
      setGroups(state.groups || []);
      setShapes(state.shapes || []);
      setTextNodes(state.textNodes || []);
      nextNodeId.current = state.nextNodeId || 0;
      nextConnectionId.current = state.nextConnectionId || 0;
      nextGroupId.current = state.nextGroupId || 0;
      nextShapeId.current = state.nextShapeId || 0;
      nextTextNodeId.current = state.nextTextNodeId || 0;
      setSelection([]);
  }, []);
  
  // -- Initialization Logic --
  useEffect(() => {
      // Load AWS Template on startup if empty
      if (nodes.length === 0 && appMode === 'aws') {
          const template = ARCHITECTURE_TEMPLATES.find(t => t.name === 'Comprehensive Cloud Architecture');
          if (template) {
              applyTemplate(template, {x: 100, y: 100});
          }
      }
  }, []); // Run once on mount

  // --- WORKFLOW ENGINE ---

  // Helper to zoom and pan to a specific node, accounting for sidebars to center perfectly
  const focusOnNode = useCallback((nodeId: string) => {
      const node = nodes.find(n => n.id === nodeId);
      if (!node) return;

      const isWf = node.type === 'start' || node.type === 'end';
      const w = isWf ? WORKFLOW_NODE_SIZE : NODE_WIDTH;
      const h = isWf ? WORKFLOW_NODE_SIZE : NODE_HEIGHT;

      // Center of the node
      const nodeCenterX = node.position.x + w / 2;
      const nodeCenterY = node.position.y + h / 2;

      const screenWidth = window.innerWidth;
      const screenHeight = window.innerHeight;

      // Desired Zoom level - Adjusted to cover about 5 elements (~0.8)
      const targetZoom = 0.8;

      // Sidebar offsets calculation
      const leftSidebarWidth = isLeftSidebarOpen ? 256 : 0; 
      // Right sidebar is 384px (w-96) normally, or 600px if in compare mode (though compare mode might be off during workflow)
      const rightSidebarWidth = isRightSidebarOpen ? (compareMode ? 600 : 384) : 0;

      // The available visual center is offset by the left sidebar
      const visualCenterX = leftSidebarWidth + (screenWidth - leftSidebarWidth - rightSidebarWidth) / 2;
      const visualCenterY = screenHeight / 2;

      // Calculate translation to center the node in the visible area
      // WorldX * Scale + TranslateX = ScreenX
      // TranslateX = ScreenX - WorldX * Scale
      const newX = visualCenterX - (nodeCenterX * targetZoom);
      const newY = visualCenterY - (nodeCenterY * targetZoom);

      setTransform({ x: newX, y: newY, k: targetZoom });
  }, [nodes, isLeftSidebarOpen, isRightSidebarOpen, compareMode]);

  // Helper to Zoom Out to Fit All
  const zoomToFit = useCallback(() => {
      if (nodes.length === 0) return;

      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      nodes.forEach(n => {
          const isWf = n.type === 'start' || n.type === 'end';
          const w = isWf ? WORKFLOW_NODE_SIZE : NODE_WIDTH;
          const h = isWf ? WORKFLOW_NODE_SIZE : NODE_HEIGHT;
          minX = Math.min(minX, n.position.x);
          minY = Math.min(minY, n.position.y);
          maxX = Math.max(maxX, n.position.x + w);
          maxY = Math.max(maxY, n.position.y + h);
      });

      // Add padding
      const padding = 100;
      const contentWidth = (maxX - minX) + padding * 2;
      const contentHeight = (maxY - minY) + padding * 2;

      // Approximate viewport
      const screenWidth = window.innerWidth; 
      const screenHeight = window.innerHeight;

      const scaleX = screenWidth / contentWidth;
      const scaleY = screenHeight / contentHeight;
      let newK = Math.min(scaleX, scaleY, 1); 
      newK = Math.max(0.1, newK); 

      // Center the content bounds
      const contentCenterX = minX + (maxX - minX) / 2;
      const contentCenterY = minY + (maxY - minY) / 2;

      const newX = (screenWidth / 2) - (contentCenterX * newK);
      const newY = (screenHeight / 2) - (contentCenterY * newK);

      setTransform({ x: newX, y: newY, k: newK });
  }, [nodes]);

  // Workflow Sequencer Effect
  useEffect(() => {
      if (workflowStatus !== 'running') {
          if (workflowTimeoutRef.current) {
              window.clearTimeout(workflowTimeoutRef.current);
              workflowTimeoutRef.current = null;
          }
          return;
      }

      // Initial Start (Step 0)
      if (currentStepIndex === 0) {
          const step = workflowQueue[0];
          if (step && step.type === 'node') {
              // Longer initial transition for smoothness
              setCameraTransitionDuration(2500); 
              focusOnNode(step.id);
              onSelect({ type: 'node', id: step.id }, false);
              
              // Find outgoing connections from start node to glow them early
              const outgoing = connections.filter(c => c.fromNodeId === step.id);
              const outgoingIds = outgoing.map(c => c.id);
              
              setAnimationState({
                  activeNodes: new Set([step.id]),
                  activeConnections: new Set(outgoingIds), // Glow initial arrows immediately
                  alertConnections: new Set()
              });

              // Wait longer on start for user orientation
              workflowTimeoutRef.current = window.setTimeout(() => {
                  setCurrentStepIndex(1);
              }, 4500);
          }
          return;
      }

      // Completion
      if (currentStepIndex >= workflowQueue.length) {
          setCameraTransitionDuration(2500); // Smooth Zoom Out
          zoomToFit();
          setWorkflowStatus('completed');
          addToast('Workflow Completed', 'bg-green-600');
          // KEEP activeConnections, clear activeNodes
          setAnimationState(prev => ({ activeNodes: new Set(), activeConnections: prev.activeConnections, alertConnections: new Set() }));
          
          // Auto-Close Sidebar on completion
          setIsRightSidebarOpen(false);
          setCompareMode(false);
          setCompareNodeId(null);

          // Reset duration and state after animation completes
          workflowTimeoutRef.current = window.setTimeout(() => {
               setCameraTransitionDuration(0); 
               setWorkflowRunningNodeId(null);
               setCurrentStepIndex(0);
          }, 2500);
          return;
      }

      const step = workflowQueue[currentStepIndex];
      let delay = 0;

      if (step.type === 'node') {
          // Standard step transition
          setCameraTransitionDuration(1500); 
          focusOnNode(step.id);
          
          const currentNode = nodes.find(n => n.id === step.id);
          const hasCode = currentNode?.data.code || currentNode?.data.codeTF;

          // ML/DL Code Comparison Logic
          if ((appMode === 'ai' || appMode === 'dl') && hasCode) {
              const history = workflowCodeHistory.current;
              const prevNodeId = history.length > 0 ? history[history.length - 1] : null;

              if (prevNodeId) {
                  // Previous Node found: Set it as Selection (Left/Base), Current as Compare (Right)
                  onSelect({ type: 'node', id: prevNodeId }, false); 
                  setCompareNodeId(step.id);
                  setCompareMode(true);
              } else {
                  // First node with code: Just select it
                  onSelect({ type: 'node', id: step.id }, false);
                  setCompareNodeId(null);
                  setCompareMode(false);
              }
              
              // Add current to history if it's new
              if (!history.includes(step.id)) {
                  history.push(step.id);
              }
          } else {
              // Standard selection for AWS or non-code nodes
              onSelect({ type: 'node', id: step.id }, false);
              setCompareMode(false);
          }
          
          // Find ALL outgoing connections from this node to light them up simultaneously
          // This creates the "branching flow" effect
          const outgoing = connections.filter(c => c.fromNodeId === step.id);
          const outgoingIds = outgoing.map(c => c.id);

          setAnimationState(prev => ({
              ...prev,
              activeNodes: new Set([step.id]), // Only highlight current node
              activeConnections: new Set([...prev.activeConnections, ...outgoingIds]) // Accumulate trails + Light up ALL outgoing
          }));
          
          // Dwell time on node
          delay = 3000;
          
      } else if (step.type === 'connection') {
          // Connection step - mainly for timing, the visual might already be lit from the previous node step
          // Ensure this specific connection is definitely in the set (redundancy is fine)
          setAnimationState(prev => {
              const nextConns = new Set(prev.activeConnections);
              nextConns.add(step.id);
              return { ...prev, activeConnections: nextConns };
          });
          delay = 800; // Shorter dwell on connection traversal
      }

      // Schedule Next Step
      workflowTimeoutRef.current = window.setTimeout(() => {
          setCurrentStepIndex(prev => prev + 1);
      }, delay);

      return () => {
          if (workflowTimeoutRef.current) {
              window.clearTimeout(workflowTimeoutRef.current);
          }
      };

  }, [workflowStatus, currentStepIndex, workflowQueue, focusOnNode, zoomToFit, addToast, connections, appMode, nodes]); 

  const generateWorkflowQueue = (startNodeId: string): WorkflowStep[] => {
      const queue: WorkflowStep[] = [];
      const visited = new Set<string>();
      const nodeMap = new Map(nodes.map(n => [n.id, n]));
      
      const linearizedSteps: WorkflowStep[] = [];
      
      const traverse = (currentId: string) => {
          linearizedSteps.push({ type: 'node', id: currentId });
          
          // Find outgoing connections
          const outgoing = connections.filter(c => c.fromNodeId === currentId);
          
          // Sort outgoing by y position to give a deterministic top-down flow
          outgoing.sort((a, b) => {
              const nodeA = nodeMap.get(a.toNodeId);
              const nodeB = nodeMap.get(b.toNodeId);
              return (nodeA?.position.y || 0) - (nodeB?.position.y || 0);
          });

          for (const conn of outgoing) {
              if (!visited.has(conn.toNodeId)) {
                  visited.add(conn.toNodeId);
                  linearizedSteps.push({ type: 'connection', id: conn.id });
                  traverse(conn.toNodeId);
              }
          }
      };
      
      traverse(startNodeId);
      return linearizedSteps;
  };

  const handleStartWorkflowWrapper = useCallback((nodeId: string) => {
      const queue = generateWorkflowQueue(nodeId);
      if (queue.length === 0) return;

      // Initial Camera Setup for Start
      setCameraTransitionDuration(2500); // Slower initial move
      focusOnNode(nodeId);

      // Reset state
      setWorkflowQueue(queue);
      setCurrentStepIndex(0);
      setWorkflowStatus('running');
      setWorkflowRunningNodeId(nodeId);
      workflowCodeHistory.current = []; // Clear code history
      
      // Open Right Sidebar, Close Left Sidebar for clean view
      setIsRightSidebarOpen(true);
      setIsLeftSidebarOpen(false);
      
      setAnimationState({ activeNodes: new Set(), activeConnections: new Set(), alertConnections: new Set() });
      addToast('Workflow Started', 'bg-blue-600');

  }, [nodes, connections, addToast, focusOnNode]); 

  const handleToggleWorkflow = useCallback(() => {
      if (workflowStatus === 'running') {
          setWorkflowStatus('paused');
          addToast('Workflow Paused', 'bg-yellow-500');
      } else if (workflowStatus === 'paused') {
          setWorkflowStatus('running');
          addToast('Workflow Resumed', 'bg-green-500');
      } else if (workflowStatus === 'completed' || workflowStatus === 'idle') {
          const startNode = nodes.find(n => n.type === 'start') || nodes[0];
          if (startNode) {
              handleStartWorkflowWrapper(startNode.id);
          }
      }
  }, [workflowStatus, nodes, handleStartWorkflowWrapper, addToast]);

  const handleStopWorkflow = useCallback(() => {
     setWorkflowStatus('idle');
     setCurrentStepIndex(0);
     setWorkflowRunningNodeId(null);
     setCameraTransitionDuration(2000); // 2s Smooth Zoom Out on Stop
     zoomToFit();
     // Keep persistent arrows on manual stop as well? 
     // Usually stop implies "reset". The prompt says "after workflow is zoomed out... keep glowing".
     // That refers to completion. For manual stop, let's reset to clean slate as per typical UX.
     setAnimationState({ activeNodes: new Set(), activeConnections: new Set(), alertConnections: new Set() });
     
     // Close property panel on stop
     setIsRightSidebarOpen(false);
     setCompareMode(false);
     setCompareNodeId(null);

     if (workflowTimeoutRef.current) clearTimeout(workflowTimeoutRef.current);
     addToast('Workflow Stopped', 'bg-gray-600');
     
     // Reset duration after zoom out
     setTimeout(() => setCameraTransitionDuration(0), 2000);
  }, [addToast, zoomToFit]);


  const startTutorialSequence = (mode: 'ai' | 'dl', loadedNodes: Node[]) => {
      setTutorialState({ active: false, step: 0, highlightNodeId: null, glowButton: null, message: null });
      
      let targetNodeId: string | undefined;
      
      if (mode === 'ai') {
          targetNodeId = loadedNodes.find(n => n.type === 'reg-linear')?.id;
          if (targetNodeId) {
              setTutorialState({ active: true, step: 1, highlightNodeId: targetNodeId, glowButton: null, message: null });
              setTimeout(() => {
                  setTutorialState(prev => prev.step === 1 ? { ...prev, step: 2 } : prev);
              }, 5000);
          }
      } else if (mode === 'dl') {
          targetNodeId = loadedNodes.find(n => n.type === 'vit-base')?.id;
          if(targetNodeId) {
              setTutorialState({ active: true, step: 1, highlightNodeId: targetNodeId, glowButton: null, message: null });
          }
      }
  }

  const handleModeChange = useCallback((newMode: 'aws' | 'ai' | 'dl') => {
      if (appMode === newMode) return;
      const currentState = getCurrentState();
      
      if (appMode === 'aws') setAwsState(currentState);
      else if (appMode === 'ai') setAiState(currentState);
      else if (appMode === 'dl') setDlState(currentState);
      
      setAppMode(newMode);
      setSelection([]);
      setUndoStack([]);
      setRedoStack([]);
      setCompareMode(false);
      setCompareNodeId(null);
      setTutorialState({ active: false, step: 0, highlightNodeId: null, glowButton: null, message: null });

      if (newMode === 'ai') {
          if (aiState) {
              loadState(aiState);
          } else {
              resetState();
              const template = AI_TEMPLATES.find(t => t.name === 'Linear Regression Pipeline');
              if (template) {
                  const { nodes: newNodes } = applyTemplateSync(template, {x: 100, y: 100});
                  startTutorialSequence('ai', newNodes);
              }
          }
          addToast("Switched to AI Visualizer", 'bg-blue-600');
          setIsLeftSidebarOpen(true);
          setIsRightSidebarOpen(false); 
      } else if (newMode === 'dl') {
          if (dlState) {
              loadState(dlState);
          } else {
              resetState();
              const template = AI_TEMPLATES.find(t => t.name === 'Vision Transformer (ViT) Classifier');
              if(template) {
                  const { nodes: newNodes } = applyTemplateSync(template, {x: 100, y: 100});
                  startTutorialSequence('dl', newNodes);
              }
          }
          addToast("Switched to Deep Learning Visualizer", 'bg-purple-600');
          setIsLeftSidebarOpen(true);
          setIsRightSidebarOpen(false);
      } else {
          if (awsState) loadState(awsState);
          else {
              resetState();
              const template = ARCHITECTURE_TEMPLATES.find(t => t.name === 'Comprehensive Cloud Architecture');
              if (template) applyTemplate(template, {x: 100, y: 100});
          }
          addToast("Switched to AWS Visualizer", 'bg-orange-600');
          setIsLeftSidebarOpen(true);
          setIsRightSidebarOpen(true);
      }
  }, [appMode, getCurrentState, loadState, awsState, aiState, dlState]);

  const resetState = () => {
      setNodes([]);
      setConnections([]);
      setGroups([]);
      setShapes([]);
      setTextNodes([]);
      nextNodeId.current = 0;
      nextConnectionId.current = 0;
      nextGroupId.current = 0;
      nextShapeId.current = 0;
      nextTextNodeId.current = 0;
  };

  useEffect(() => {
      if (appMode === 'aws') {
          const issues = validateGraph(nodes, connections);
          setValidationIssues(issues);
          if (selection.length === 1 && selection[0].type === 'node') {
              const issue = issues.find(i => i.nodeId === selection[0].id);
              setSelectedIssue(issue || null);
          } else {
              setSelectedIssue(null);
          }
      } else {
          setValidationIssues([]);
          setSelectedIssue(null);
      }
  }, [nodes, connections, selection, appMode]);

  const pushToUndoStack = useCallback(() => {
    setUndoStack(prev => [...prev.slice(-HISTORY_LIMIT + 1), getCurrentState()]);
    setRedoStack([]);
  }, [getCurrentState]);

  const handleUndo = useCallback(() => {
    if (undoStack.length === 0) return;
    const lastState = undoStack[undoStack.length - 1];
    setRedoStack(prev => [...prev, getCurrentState()]);
    setUndoStack(prev => prev.slice(0, -1));
    loadState(lastState);
  }, [undoStack, getCurrentState, loadState]);

  const handleRedo = useCallback(() => {
    if (redoStack.length === 0) return;
    const nextState = redoStack[redoStack.length - 1];
    setUndoStack(prev => [...prev, getCurrentState()]);
    setRedoStack(prev => prev.slice(0, -1));
    loadState(nextState);
  }, [redoStack, getCurrentState, loadState]);

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [theme]);

  useEffect(() => {
    const savedStateJSON = localStorage.getItem('aws-designer-state');
    if (savedStateJSON) {
        try {
            const savedState = JSON.parse(savedStateJSON);
            loadState(savedState as AppState);
            setUndoStack([]);
            setRedoStack([]);
        } catch (e) {
            console.error("Failed to parse saved state:", e);
        }
    }
    const savedTheme = localStorage.getItem('aws-designer-theme');
    if (savedTheme === 'light' || savedTheme === 'dark') {
        setTheme(savedTheme);
    }
  }, [loadState]);

  useEffect(() => {
      if (appMode === 'aws') {
          localStorage.setItem('aws-designer-state', JSON.stringify(getCurrentState()));
      }
  }, [nodes, connections, groups, shapes, textNodes, getCurrentState, appMode]);
  
  useEffect(() => {
      localStorage.setItem('aws-designer-theme', theme);
  }, [theme]);
  
  const handleAutoFix = useCallback((issue: ValidationIssue) => {
      if (!issue.fixAction || !issue.missingComponent) return;
      const targetNode = nodes.find(n => n.id === issue.nodeId);
      if (!targetNode) return;
      const service = ALL_SERVICES.find(s => s.id === issue.fixAction?.nodeType);
      if (!service) return;

      pushToUndoStack();
      const INSERTION_GAP = NODE_WIDTH + 100;
      const isBefore = issue.fixAction.position === 'before';
      const newPos = { ...targetNode.position };
      if (!isBefore) newPos.x += INSERTION_GAP; 
      
      const shiftThreshold = isBefore ? targetNode.position.x - 10 : targetNode.position.x + NODE_WIDTH + 10;
      const shiftedNodes = nodes.map(n => {
          if (n.position.x >= shiftThreshold) return { ...n, position: { ...n.position, x: n.position.x + INSERTION_GAP } };
          return n;
      });

      const newNode: Node = {
          id: `node-${nextNodeId.current++}`,
          type: service.id,
          position: isBefore ? targetNode.position : newPos,
          data: { ...service.defaultData },
          isAnimating: false
      };
      
      const finalNodes = [...shiftedNodes, newNode];
      setNodes(finalNodes);
      
      let updatedConnections = [...connections];
      if (isBefore) {
          updatedConnections = updatedConnections.map(c => c.toNodeId === targetNode.id ? { ...c, toNodeId: newNode.id } : c);
          updatedConnections.push({ id: `conn-${nextConnectionId.current++}`, fromNodeId: newNode.id, toNodeId: targetNode.id, isAnimating: false });
      } else {
          updatedConnections = updatedConnections.map(c => c.fromNodeId === targetNode.id ? { ...c, fromNodeId: newNode.id } : c);
          updatedConnections.push({ id: `conn-${nextConnectionId.current++}`, fromNodeId: targetNode.id, toNodeId: newNode.id, isAnimating: false });
      }

      setConnections(updatedConnections);
      addToast(`Fixed: Added ${service.name}`, 'bg-blue-600');
      setSelectedIssue(null);
  }, [nodes, connections, pushToUndoStack, addToast]);

  const addNode = useCallback((service: AWSService, position: Vector2D, dataOverrides: Partial<NodeData> = {}): Node => {
    pushToUndoStack();
    const newNode: Node = {
      id: `node-${nextNodeId.current++}`,
      type: service.id,
      position,
      data: { ...service.defaultData, ...dataOverrides },
      isAnimating: false,
    };
    setNodes((prevNodes) => [...prevNodes, newNode]);
    return newNode;
  }, [pushToUndoStack]);

  const addGroup = useCallback((position: Vector2D) => {
      pushToUndoStack();
      const newGroup: Group = {
          id: `group-${nextGroupId.current++}`,
          type: 'group',
          position,
          size: { width: 400, height: 300 },
          data: {
              label: 'New Group',
              borderStyle: 'dotted',
              borderColor: '#6b7280',
              backgroundColor: 'rgba(75, 85, 99, 0.1)',
              locked: false,
          },
          zIndex: 0,
      };
      setGroups(prev => [...prev, newGroup]);
      return newGroup;
  }, [pushToUndoStack]);

  const addShape = useCallback((shapeType: 'rectangle' | 'ellipse', position: Vector2D) => {
    pushToUndoStack();
    const newShape: Shape = {
      id: `shape-${nextShapeId.current++}`,
      type: shapeType,
      position,
      size: { width: 200, height: 150 },
      data: {
        fillColor: 'rgba(107, 114, 128, 0.2)',
        strokeColor: '#9ca3af',
        strokeWidth: 2,
        borderStyle: 'solid',
      },
      zIndex: -1
    };
    setShapes(prev => [...prev, newShape]);
    return newShape;
  }, [pushToUndoStack]);

  const addTextNode = useCallback((position: Vector2D) => {
      pushToUndoStack();
      const newTextNode: TextNode = {
          id: `text-${nextTextNodeId.current++}`,
          type: 'text',
          position,
          size: { width: 150, height: 40 },
          data: {
              content: 'Double-click to edit',
              fontSize: 16,
              color: '#cbd5e1',
          },
          zIndex: 1,
      };
      setTextNodes(prev => [...prev, newTextNode]);
      return newTextNode;
  }, [pushToUndoStack]);

  const addConnection = useCallback((fromNodeId: string, toNodeId: string): Connection | undefined => {
    if (fromNodeId === toNodeId) return;
    const existing = connections.find(c => c.fromNodeId === fromNodeId && c.toNodeId === toNodeId);
    if (existing) return;

    pushToUndoStack();
    const newConnection: Connection = {
      id: `conn-${nextConnectionId.current++}`,
      fromNodeId,
      toNodeId,
      label: '',
      isAnimating: false,
    };
    setConnections((prev) => [...prev, newConnection]);
    return newConnection;
  }, [connections, pushToUndoStack]);

  const calculateSmartPosition = useCallback((existingNodes: Node[]) => {
      if (existingNodes.length === 0) return { x: 100, y: 100 };
      
      let maxX = -Infinity;
      let minY = Infinity;
      
      existingNodes.forEach(n => {
          const w = (n.type === 'start' || n.type === 'end') ? WORKFLOW_NODE_SIZE : NODE_WIDTH;
          maxX = Math.max(maxX, n.position.x + w);
          minY = Math.min(minY, n.position.y);
      });
      
      return { x: maxX + 150, y: Math.max(50, minY) };
  }, []);

  const applyTemplate = useCallback((template: Template, position: Vector2D) => {
      pushToUndoStack();
      const result = applyTemplateSync(template, position);
      setNodes(prev => [...prev, ...result.nodes]);
      setConnections(prev => [...prev, ...result.connections]);
      addToast(`Applied '${template.name}' template!`);
  }, [pushToUndoStack, addToast, nodes, calculateSmartPosition]);

  const applyTemplateSync = useCallback((template: Template, position: Vector2D) => {
      const smartPos = calculateSmartPosition(nodes);
      const startX = (position.x === 100 && position.y === 100) ? smartPos.x : position.x;
      const startY = (position.x === 100 && position.y === 100) ? smartPos.y : position.y;

      const nodeMap = new Map<string, string>();
      const newNodes: Node[] = [];
      template.nodes.forEach(templateNode => {
          const service = ALL_SERVICES.find(s => s.id === templateNode.type);
          if (service) {
              const newNode: Node = {
                id: `node-${nextNodeId.current++}`,
                type: service.id,
                position: {
                    x: startX + (templateNode.position.x || 0),
                    y: startY + (templateNode.position.y || 0),
                },
                data: { ...service.defaultData, ...templateNode.data },
                isAnimating: false,
              };
              newNodes.push(newNode);
              nodeMap.set(templateNode.id, newNode.id);
          }
      });
      
      const newConnections: Connection[] = [];
      template.connections.forEach(templateConn => {
          const fromId = nodeMap.get(templateConn.from);
          const toId = nodeMap.get(templateConn.to);
          if (fromId && toId) {
            const newConnection: Connection = {
                id: `conn-${nextConnectionId.current++}`,
                fromNodeId: fromId,
                toNodeId: toId,
                label: '',
                isAnimating: false,
            };
            newConnections.push(newConnection);
          }
      });
      
      return { nodes: newNodes, connections: newConnections };
  }, [nodes, calculateSmartPosition]);

  const handleAddCategory = useCallback((category: AWSCategory) => {
      pushToUndoStack();
      const smartPos = calculateSmartPosition(nodes);
      const COLUMNS = Math.ceil(Math.sqrt(category.services.length));
      const X_GAP = 220;
      const Y_GAP = 120;

      const newNodes: Node[] = category.services.map((service, index) => {
          const col = index % COLUMNS;
          const row = Math.floor(index / COLUMNS);
          
          return {
              id: `node-${nextNodeId.current++}`,
              type: service.id,
              position: {
                  x: smartPos.x + (col * X_GAP),
                  y: smartPos.y + (row * Y_GAP)
              },
              data: { ...service.defaultData },
              isAnimating: false
          }
      });

      setNodes(prev => [...prev, ...newNodes]);
      addToast(`Added all ${category.name} components`, 'bg-blue-600');
  }, [pushToUndoStack, calculateSmartPosition, nodes, addToast]);


  const addRelatedNode = useCallback((sourceNode: Node, service: AWSService) => {
      pushToUndoStack();
      const isSourceWorkflow = sourceNode.type === 'start' || sourceNode.type === 'end';
      const sourceSize = isSourceWorkflow ? {w: WORKFLOW_NODE_SIZE, h: WORKFLOW_NODE_SIZE} : {w: NODE_WIDTH, h: NODE_HEIGHT};
      
      let position: Vector2D;
      
      if (sourceNode.type === 'tg') {
          // TARGET GROUP GRID LAYOUT LOGIC
          // Find existing children loosely based on position containment
          // Assuming TG is 300x200 (hardcoded in DraggableNode)
          const tgWidth = 300;
          const tgHeight = 200;
          
          const children = nodes.filter(n => 
              n.position.x >= sourceNode.position.x && 
              n.position.x < sourceNode.position.x + tgWidth &&
              n.position.y >= sourceNode.position.y &&
              n.position.y < sourceNode.position.y + tgHeight
          );
          
          const count = children.length;
          const col = count % 2; // 2 columns
          const row = Math.floor(count / 2);
          
          // Calculate precise grid position
          position = { 
              x: sourceNode.position.x + 20 + (col * (NODE_WIDTH - 40)), // Overlap slightly or tight fit
              y: sourceNode.position.y + 40 + (row * (NODE_HEIGHT + 10))
          };
          
      } else {
          // Default linear placement
          position = { x: sourceNode.position.x + sourceSize.w + 100, y: sourceNode.position.y };
      }

      const newNode: Node = {
        id: `node-${nextNodeId.current++}`,
        type: service.id,
        position,
        data: { ...service.defaultData },
        isAnimating: false,
      };
      setNodes((prevNodes) => [...prevNodes, newNode]);
      
      if (sourceNode.type !== 'tg') {
        const newConnection: Connection = {
            id: `conn-${nextConnectionId.current++}`,
            fromNodeId: sourceNode.id,
            toNodeId: newNode.id,
            label: '',
            isAnimating: false,
        };
        setConnections((prev) => [...prev, newConnection]);
      }
  }, [pushToUndoStack, nodes]);
  
  const updateNodeGroup = useCallback((nodeId: string) => {
      setNodes(prevNodes => {
          const node = prevNodes.find(n => n.id === nodeId);
          if (!node) return prevNodes;
          const isWorkflow = node.type === 'start' || node.type === 'end';
          const size = isWorkflow ? {w: WORKFLOW_NODE_SIZE, h: WORKFLOW_NODE_SIZE} : {w: NODE_WIDTH, h: NODE_HEIGHT};
          const nodeCenter = { x: node.position.x + size.w/2, y: node.position.y + size.h/2 };
          
          const potentialParents = groups.filter(g => 
              nodeCenter.x >= g.position.x && nodeCenter.x <= g.position.x + g.size.width &&
              nodeCenter.y >= g.position.y && nodeCenter.y <= g.position.y + g.size.height
          );

          if (potentialParents.length > 0) {
              const smallestParent = potentialParents.reduce((smallest, current) => 
                  (current.size.width * current.size.height < smallest.size.width * smallest.size.height) ? current : smallest
              );
              return prevNodes.map(n => n.id === nodeId ? { ...n, groupId: smallestParent.id } : n);
          } else {
              return prevNodes.map(n => n.id === nodeId ? { ...n, groupId: undefined } : n);
          }
      });
  }, [groups]);

  const updateNodePosition = useCallback((nodeId: string, newPosition: Vector2D) => {
    setNodes((prevNodes) => prevNodes.map((node) => node.id === nodeId ? { ...node, position: newPosition } : node));
    updateNodeGroup(nodeId);
  }, [updateNodeGroup]);
  
  const updateNodeData = useCallback((nodeId: string, newData: Partial<NodeData>) => {
    pushToUndoStack();
    setNodes(prevNodes => {
        const targetNode = prevNodes.find(n => n.id === nodeId);
        if(!targetNode) return prevNodes;
        let finalData = { ...targetNode.data, ...newData };
        
        if (targetNode.type === 'ec2' && (newData.instanceFamily || newData.instanceSize)) {
            if (targetNode.data.health !== 'healthy') {
                finalData.health = 'healthy';
                addToast('EC2 Scaled Up: Health restored & System traffic normalized', 'bg-blue-500');
            }
        }
        
        return prevNodes.map(node => {
            if (node.id === nodeId) {
                if(node.type === 'asg') {
                    const { minSize, maxSize, desiredCapacity } = finalData;
                    if (parseInt(desiredCapacity) < parseInt(minSize)) finalData.health = 'warning';
                    else if (parseInt(desiredCapacity) > parseInt(maxSize)) finalData.health = 'critical';
                    else finalData.health = 'healthy';
                }
                return { ...node, data: finalData };
            }
            if (targetNode.type === 'ec2' && finalData.health === 'healthy' && targetNode.data.health !== 'healthy') {
                 if (node.type.startsWith('bu-')) return { ...node, data: { ...node.data, traffic: 'normal' as TrafficLevel } };
            }
            return node;
        });
    });
  }, [pushToUndoStack, addToast]);

  const deleteNode = useCallback((nodeId: string) => {
      pushToUndoStack();
      setNodes(nodes => nodes.filter(n => n.id !== nodeId));
      setConnections(conns => conns.filter(c => c.fromNodeId !== nodeId && c.toNodeId !== nodeId));
      setSelection(prev => prev.filter(s => s.id !== nodeId));
  }, [pushToUndoStack]);
  
  const disconnectNode = useCallback((nodeId: string) => {
      pushToUndoStack();
      setConnections(conns => conns.filter(c => c.fromNodeId !== nodeId && c.toNodeId !== nodeId));
      addToast('Connections removed');
  }, [pushToUndoStack, addToast]);

  const updateGroup = useCallback((groupId: string, data: Partial<GroupData> & { position?: Vector2D, size?: {width: number, height: number} }) => {
    setGroups(prevGroups => prevGroups.map(group => group.id === groupId ? { ...group, position: data.position || group.position, size: data.size || group.size, data: { ...group.data, ...data } } : group));
  }, []);
  
  const updateGroupData = useCallback((groupId: string, data: Partial<GroupData>) => {
    pushToUndoStack();
    updateGroup(groupId, data);
  }, [pushToUndoStack, updateGroup])

  const updateGroupPosition = useCallback((groupId: string, delta: Vector2D) => {
      setGroups(prev => prev.map(g => g.id === groupId ? {...g, position: {x: g.position.x + delta.x, y: g.position.y + delta.y}} : g));
      setNodes(prev => prev.map(n => n.groupId === groupId ? {...n, position: {x: n.position.x + delta.x, y: n.position.y + delta.y}} : n));
  }, []);

  const deleteGroup = useCallback((groupId: string) => {
    pushToUndoStack();
    setGroups(groups => groups.filter(g => g.id !== groupId));
    setNodes(nodes => nodes.map(n => n.groupId === groupId ? { ...n, groupId: undefined } : n));
    setSelection(prev => prev.filter(s => s.id !== groupId));
  }, [pushToUndoStack]);

  const updateShape = useCallback((shapeId: string, data: Partial<ShapeData> & { position?: Vector2D, size?: {width: number, height: number} }) => {
      setShapes(prev => prev.map(shape => shape.id === shapeId ? { ...shape, position: data.position || shape.position, size: data.size || shape.size, data: { ...shape.data, ...data } } : shape));
  }, []);

  const updateShapeData = useCallback((shapeId: string, data: Partial<ShapeData>) => {
    pushToUndoStack();
    updateShape(shapeId, data);
  }, [pushToUndoStack, updateShape]);

  const deleteShape = useCallback((shapeId: string) => {
      pushToUndoStack();
      setShapes(prev => prev.filter(s => s.id !== shapeId));
      setSelection(prev => prev.filter(s => s.id !== shapeId));
  }, [pushToUndoStack]);

  const updateTextNode = useCallback((textId: string, data: Partial<TextNodeData> & { position?: Vector2D, size?: {width: number, height: number} }) => {
      setTextNodes(prev => prev.map(textNode => textNode.id === textId ? { ...textNode, position: data.position || textNode.position, size: data.size || textNode.size, data: { ...textNode.data, ...data } } : textNode));
  }, []);

  const updateTextNodeData = useCallback((textId: string, data: Partial<TextNodeData>) => {
      pushToUndoStack();
      updateTextNode(textId, data);
  }, [pushToUndoStack, updateTextNode]);

  const deleteTextNode = useCallback((textId: string) => {
      pushToUndoStack();
      setTextNodes(prev => prev.filter(t => t.id !== textId));
      setSelection(prev => prev.filter(s => s.id !== textId));
  }, [pushToUndoStack]);

  const updateConnectionData = useCallback((connId: string, newData: Partial<Connection>) => {
    pushToUndoStack();
    setConnections(prev => prev.map(c => c.id === connId ? { ...c, ...newData } : c));
  }, [pushToUndoStack]);

  const deleteConnection = useCallback((connId: string) => {
      pushToUndoStack();
      setConnections(conns => conns.filter(c => c.id !== connId));
      setSelection(prev => prev.filter(s => s.id !== connId));
  }, [pushToUndoStack]);
  
  const handleErase = useCallback(() => {
    pushToUndoStack();
    setNodes([]);
    setConnections([]);
    setGroups([]);
    setShapes([]);
    setTextNodes([]);
    nextNodeId.current = 0;
    nextConnectionId.current = 0;
    nextGroupId.current = 0;
    nextShapeId.current = 0;
    nextTextNodeId.current = 0;
    addToast('Canvas cleared!', 'bg-blue-500');
  }, [pushToUndoStack, addToast]);

  const handleManualTransformChange: React.Dispatch<React.SetStateAction<{ x: number; y: number; k: number; }>> = (value) => {
      // If manual interaction occurs, ensure we aren't in a transition state
      if (cameraTransitionDuration !== 0) {
          setCameraTransitionDuration(0);
      }
      setTransform(value);
  };

  const handleZoom = (amount: number) => {
      setCameraTransitionDuration(0);
      setTransform(prev => {
          const newK = Math.max(0.1, Math.min(5, prev.k + amount));
          return { ...prev, k: newK };
      });
  }
  
  const setZoomLevel = (percent: number) => {
       setCameraTransitionDuration(0);
       setTransform(prev => {
          const newK = Math.max(0.1, Math.min(5, percent / 100));
          return { ...prev, k: newK };
      });
  }


  const onSelect = (item: Selection | null, additive: boolean = false) => {
    if (item === null) {
        setSelection([]);
        return;
    }
    
    if ((appMode === 'ai' || appMode === 'dl') && item.type === 'node') {
        setIsRightSidebarOpen(true);
        if (compareMode) {
            if (selection.length > 0 && selection[0].id !== item.id) {
                setCompareNodeId(item.id);
                return;
            }
        }
    }
    
    if (tutorialState.active && tutorialState.step === 1 && item.id === tutorialState.highlightNodeId) {
        setTutorialState(prev => ({ 
            ...prev, 
            step: 2, 
            highlightNodeId: null, 
            glowButton: appMode === 'dl' ? 'tensorflow' : 'compare',
            message: 'Now check the inspector!'
        }));
        
        setTimeout(() => {
             setTutorialState(prev => prev.step === 2 ? {
                 ...prev,
                 step: 3,
                 message: 'Select another node to compare!',
                 glowButton: null
             } : prev);
        }, 1500);
    }

    if (additive) {
        setSelection(currentSelection => {
            const isSelected = currentSelection.some(s => s.id === item.id && s.type === item.type);
            if (isSelected) {
                return currentSelection.filter(s => s.id !== item.id || s.type !== item.type);
            } else {
                return [...currentSelection, item];
            }
        });
    } else {
        const isSelected = selection.some(s => s.id === item.id && s.type === item.type);
        if (selection.length === 1 && isSelected) {
            if ((appMode === 'ai' || appMode === 'dl') && !isRightSidebarOpen) {
                setIsRightSidebarOpen(true);
            }
            return;
        }
        setSelection([item]);
    }
  };

  const handleMultiSelect = (rect: {x: number, y: number, width: number, height: number}) => {
      const selectedIds: Selection[] = [];
      nodes.forEach(n => {
          const isWf = n.type === 'start' || n.type === 'end';
          const w = isWf ? WORKFLOW_NODE_SIZE : NODE_WIDTH;
          const h = isWf ? WORKFLOW_NODE_SIZE : NODE_HEIGHT;
          if (n.position.x < rect.x + rect.width && n.position.x + w > rect.x &&
              n.position.y < rect.y + rect.height && n.position.y + h > rect.y) {
              selectedIds.push({ type: 'node', id: n.id });
          }
      });
      setSelection(selectedIds);
  };

  const handleAlign = (direction: 'top' | 'middle' | 'bottom' | 'left' | 'center' | 'right') => {
    pushToUndoStack();
    const selectedNodes = selection
        .filter(s => s.type === 'node')
        .map(s => nodes.find(n => n.id === s.id))
        .filter((n): n is Node => n !== undefined);

    if (selectedNodes.length < 2) return;

    const bounds = {
        minX: Math.min(...selectedNodes.map(n => n.position.x)),
        maxX: Math.max(...selectedNodes.map(n => {
             const w = (n.type === 'start' || n.type === 'end') ? WORKFLOW_NODE_SIZE : NODE_WIDTH;
             return n.position.x + w;
        })),
        minY: Math.min(...selectedNodes.map(n => n.position.y)),
        maxY: Math.max(...selectedNodes.map(n => {
             const h = (n.type === 'start' || n.type === 'end') ? WORKFLOW_NODE_SIZE : NODE_HEIGHT;
             return n.position.y + h;
        })),
    };

    setNodes(currentNodes => {
        return currentNodes.map(n => {
            if (!selectedNodes.some(sn => sn.id === n.id)) return n;
            const newPosition = { ...n.position };
            const isWorkflow = n.type === 'start' || n.type === 'end';
            const w = isWorkflow ? WORKFLOW_NODE_SIZE : NODE_WIDTH;
            const h = isWorkflow ? WORKFLOW_NODE_SIZE : NODE_HEIGHT;

            switch (direction) {
                case 'left':    newPosition.x = bounds.minX; break;
                case 'center':  newPosition.x = bounds.minX + (bounds.maxX - bounds.minX) / 2 - w / 2; break;
                case 'right':   newPosition.x = bounds.maxX - w; break;
                case 'top':     newPosition.y = bounds.minY; break;
                case 'middle':  newPosition.y = bounds.minY + (bounds.maxY - bounds.minY) / 2 - h / 2; break;
                case 'bottom':  newPosition.y = bounds.maxY - h; break;
            }
            return { ...n, position: newPosition };
        });
    });
  };

  const handleDistribute = (direction: 'horizontal' | 'vertical') => {
      pushToUndoStack();
      const selectedNodes = selection
        .filter(s => s.type === 'node')
        .map(s => nodes.find(n => n.id === s.id))
        .filter((n): n is Node => n !== undefined);

      if (selectedNodes.length < 3) return; // Need 3 to distribute

      if (direction === 'horizontal') {
          selectedNodes.sort((a, b) => a.position.x - b.position.x);
          const minX = selectedNodes[0].position.x;
          const maxX = selectedNodes[selectedNodes.length - 1].position.x;
          const totalSpan = maxX - minX;
          const step = totalSpan / (selectedNodes.length - 1);
          
          setNodes(currentNodes => currentNodes.map(n => {
              const index = selectedNodes.findIndex(sn => sn.id === n.id);
              if (index === -1) return n;
              return { ...n, position: { ...n.position, x: minX + (step * index) } };
          }));
      } else {
          selectedNodes.sort((a, b) => a.position.y - b.position.y);
          const minY = selectedNodes[0].position.y;
          const maxY = selectedNodes[selectedNodes.length - 1].position.y;
          const totalSpan = maxY - minY;
          const step = totalSpan / (selectedNodes.length - 1);

          setNodes(currentNodes => currentNodes.map(n => {
              const index = selectedNodes.findIndex(sn => sn.id === n.id);
              if (index === -1) return n;
              return { ...n, position: { ...n.position, y: minY + (step * index) } };
          }));
      }
  }
  
  const handleGapDistribution = (gap: number) => {
      if (!selectionBounds) return;
      
      const selectedNodes = selection
        .filter(s => s.type === 'node')
        .map(s => nodes.find(n => n.id === s.id))
        .filter((n): n is Node => n !== undefined);
        
      if (selectedNodes.length < 2) return;

      const isHorizontal = selectionBounds.w > selectionBounds.h;
      
      if (isHorizontal) {
          // Sort by X
          selectedNodes.sort((a, b) => a.position.x - b.position.x);
          
          let currentX = selectedNodes[0].position.x;
          const updatedNodes = [...nodes];
          
          selectedNodes.forEach((node, index) => {
              if (index === 0) {
                  currentX += (node.type === 'start' || node.type === 'end' ? WORKFLOW_NODE_SIZE : NODE_WIDTH) + gap;
                  return; // Reference node stays put
              }
              
              const nodeIndex = updatedNodes.findIndex(n => n.id === node.id);
              if (nodeIndex !== -1) {
                  updatedNodes[nodeIndex] = {
                      ...updatedNodes[nodeIndex],
                      position: { ...updatedNodes[nodeIndex].position, x: currentX }
                  };
                  currentX += (node.type === 'start' || node.type === 'end' ? WORKFLOW_NODE_SIZE : NODE_WIDTH) + gap;
              }
          });
          setNodes(updatedNodes);
      } else {
          // Sort by Y
          selectedNodes.sort((a, b) => a.position.y - b.position.y);
          
          let currentY = selectedNodes[0].position.y;
          const updatedNodes = [...nodes];
          
          selectedNodes.forEach((node, index) => {
              if (index === 0) {
                  currentY += (node.type === 'start' || node.type === 'end' ? WORKFLOW_NODE_SIZE : NODE_HEIGHT) + gap;
                  return; // Reference node stays put
              }
              
              const nodeIndex = updatedNodes.findIndex(n => n.id === node.id);
              if (nodeIndex !== -1) {
                  updatedNodes[nodeIndex] = {
                      ...updatedNodes[nodeIndex],
                      position: { ...updatedNodes[nodeIndex].position, y: currentY }
                  };
                  currentY += (node.type === 'start' || node.type === 'end' ? WORKFLOW_NODE_SIZE : NODE_HEIGHT) + gap;
              }
          });
          setNodes(updatedNodes);
      }
  }

  const exportAsJson = useCallback(() => {
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(getCurrentState(), null, 2));
    const downloadAnchorNode = document.createElement('a');
    downloadAnchorNode.setAttribute("href", dataStr);
    downloadAnchorNode.setAttribute("download", `aws-architecture-${Date.now()}.json`);
    document.body.appendChild(downloadAnchorNode);
    downloadAnchorNode.click();
    downloadAnchorNode.remove();
    addToast('Exported as JSON', 'bg-green-600');
  }, [getCurrentState, addToast]);

  const exportAsImage = useCallback(async (area: {x: number, y: number, width: number, height: number} | null) => {
    setIsSelectingForExport(false);
    // Short delay to allow UI to update (hide selection box)
    await new Promise(resolve => setTimeout(resolve, 50));
    
    const element = document.querySelector('main');
    if (!element) return;

    addToast('Generating Image...', 'bg-blue-500');

    try {
        const canvas = await html2canvas(element as HTMLElement, {
            useCORS: true,
            backgroundColor: theme === 'dark' ? '#111827' : '#f9fafb',
            logging: false,
            ignoreElements: (node: Element) => node.classList.contains('selection-overlay') || node.classList.contains('toast-notification')
        });

        let finalCanvas = canvas;
        if (area) {
             const screenX = area.x * transform.k + transform.x;
             const screenY = area.y * transform.k + transform.y;
             const screenW = area.width * transform.k;
             const screenH = area.height * transform.k;

             const cropped = document.createElement('canvas');
             cropped.width = screenW;
             cropped.height = screenH;
             const ctx = cropped.getContext('2d');
             if (ctx) {
                 ctx.drawImage(canvas, screenX, screenY, screenW, screenH, 0, 0, screenW, screenH);
                 finalCanvas = cropped;
             }
        }

        const link = document.createElement('a');
        link.download = `diagram-${Date.now()}.png`;
        link.href = finalCanvas.toDataURL('image/png');
        link.click();
        addToast('Image Downloaded', 'bg-green-600');
    } catch (e) {
        console.error("Export image failed", e);
        addToast('Export failed', 'bg-red-600');
    }
  }, [theme, transform, addToast]);

  const handleExportGif = useCallback(() => {
    if (isRecordingGif) return;
    const element = document.querySelector('main');
    if (!element) return;

    setIsRecordingGif(true);
    addToast('Recording GIF (Capture...', 'bg-blue-600');
    
    // We assume GIF.js is loaded
    const gif = new GIF({
      workers: 2,
      quality: 10,
      width: element.clientWidth,
      height: element.clientHeight,
      // Attempt to use a CDN worker if local is missing, or rely on setup
      workerScript: 'https://cdnjs.cloudflare.com/ajax/libs/gif.js/0.2.0/gif.worker.js' 
    });
    
    let frames = 0;
    const maxFrames = 15; // Limit frames for performance example

    const captureFrame = async () => {
        if (frames >= maxFrames) {
            addToast('Rendering GIF...', 'bg-blue-500');
            gif.on('finished', (blob: Blob) => {
                const link = document.createElement('a');
                link.href = URL.createObjectURL(blob);
                link.download = `recording-${Date.now()}.gif`;
                link.click();
                setIsRecordingGif(false);
                addToast('GIF Saved', 'bg-green-600');
            });
            gif.render();
            return;
        }

        try {
            const canvas = await html2canvas(element as HTMLElement, {
                useCORS: true,
                backgroundColor: theme === 'dark' ? '#111827' : '#f9fafb',
                logging: false,
                ignoreElements: (node: Element) => node.classList.contains('selection-overlay')
            });
            gif.addFrame(canvas, {delay: 500});
            frames++;
            setTimeout(captureFrame, 500); // 2 FPS roughly
        } catch (e) {
            console.error(e);
            setIsRecordingGif(false);
            addToast('GIF Failed', 'bg-red-600');
        }
    }
    
    captureFrame();
  }, [isRecordingGif, theme, addToast]);

  const AlignmentToolbar = () => {
      const [gap, setGap] = useState(50);
      const isDragging = useRef(false);
      const startY = useRef(0);
      const startGap = useRef(0);

      const handleScrubberDown = (e: React.MouseEvent) => {
          e.preventDefault();
          isDragging.current = true;
          startY.current = e.clientY;
          startGap.current = gap;
          pushToUndoStack(); // Save state before drag starts
          
          const handleMouseMove = (ev: MouseEvent) => {
              if (!isDragging.current) return;
              const diff = startY.current - ev.clientY; // Drag up increases gap
              const newGap = Math.max(0, startGap.current + diff);
              setGap(newGap);
              handleGapDistribution(newGap);
          };
          
          const handleMouseUp = () => {
              isDragging.current = false;
              window.removeEventListener('mousemove', handleMouseMove);
              window.removeEventListener('mouseup', handleMouseUp);
          };
          
          window.addEventListener('mousemove', handleMouseMove);
          window.addEventListener('mouseup', handleMouseUp);
      };

      if (!selectionBounds) return null;
      
      const screenX = selectionBounds.maxX * transform.k + transform.x;
      const screenY = selectionBounds.minY * transform.k + transform.y - 50; 

      return (
          <div 
            className="absolute z-50 flex items-center space-x-1 bg-white dark:bg-gray-800 p-1.5 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700/50 animate-bounce-in"
            style={{ left: screenX, top: screenY, transform: 'translateX(-100%)' }}
          >
              <button onClick={() => handleAlign('left')} title="Align Left" className="p-1.5 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"><AlignLeftIcon className="w-4 h-4"/></button>
              <button onClick={() => handleAlign('center')} title="Align Center" className="p-1.5 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"><AlignCenterIcon className="w-4 h-4"/></button>
              <button onClick={() => handleAlign('right')} title="Align Right" className="p-1.5 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"><AlignRightIcon className="w-4 h-4"/></button>
              <div className="w-px h-4 bg-gray-300 dark:bg-gray-600 mx-1"></div>
              <button onClick={() => handleAlign('top')} title="Align Top" className="p-1.5 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"><AlignTopIcon className="w-4 h-4"/></button>
              <button onClick={() => handleAlign('middle')} title="Align Middle" className="p-1.5 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"><AlignMiddleIcon className="w-4 h-4"/></button>
              <button onClick={() => handleAlign('bottom')} title="Align Bottom" className="p-1.5 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"><AlignBottomIcon className="w-4 h-4"/></button>
              <div className="w-px h-4 bg-gray-300 dark:bg-gray-600 mx-1"></div>
              <button onClick={() => handleDistribute('horizontal')} title="Tidy Up Horizontal" className="p-1.5 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"><DistributeHorizontalIcon className="w-4 h-4"/></button>
              <button onClick={() => handleDistribute('vertical')} title="Tidy Up Vertical" className="p-1.5 rounded text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"><DistributeVerticalIcon className="w-4 h-4"/></button>
              
              {/* Gap Tool */}
              <div className="w-px h-4 bg-gray-300 dark:bg-gray-600 mx-1"></div>
              <div className="flex items-center space-x-1 bg-gray-100 dark:bg-gray-900 rounded px-1">
                  <input 
                    type="number" 
                    value={Math.round(gap)} 
                    onChange={(e) => { 
                        const val = parseInt(e.target.value) || 0;
                        setGap(val);
                        handleGapDistribution(val);
                    }}
                    className="w-8 text-xs bg-transparent text-center focus:outline-none font-mono text-gray-600 dark:text-gray-300 appearance-none"
                  />
                  <div 
                    className="cursor-ns-resize text-gray-400 hover:text-orange-500 p-0.5"
                    onMouseDown={handleScrubberDown}
                    title="Drag up/down to adjust spacing"
                  >
                      <DoubleArrowIcon className="w-3 h-3 transform rotate-90" />
                  </div>
              </div>
          </div>
      )
  }

  const selectedNode = selection.length === 1 && selection[0].type === 'node' ? nodes.find(n => n.id === selection[0].id) : undefined;
  const compareNode = compareNodeId ? nodes.find(n => n.id === compareNodeId) : undefined;
  const selectedConnection = selection.length === 1 && selection[0].type === 'connection' ? connections.find(c => c.id === selection[0].id) : undefined;
  const selectedGroup = selection.length === 1 && selection[0].type === 'group' ? groups.find(g => g.id === selection[0].id) : undefined;
  const selectedShape = selection.length === 1 && selection[0].type === 'shape' ? shapes.find(s => s.id === selection[0].id) : undefined;
  const selectedTextNode = selection.length === 1 && selection[0].type === 'text' ? textNodes.find(t => t.id === selection[0].id) : undefined;
  
  const isZenMode = !isLeftSidebarOpen && !isRightSidebarOpen;

  return (
    <div className={`flex h-screen w-screen font-sans bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-200 overflow-hidden ${theme}`}>
        {toasts.map(toast => (
            <div key={toast.id} className={`toast-notification ${toast.color}`}>
                <div className="relative z-10">{toast.message}</div>
                <div className="toast-progress" style={{ animationDuration: `${toast.duration}ms` }}></div>
            </div>
        ))}
        {isCommandPaletteOpen && 
            <CommandPalette 
                onClose={() => setIsCommandPaletteOpen(false)}
                addNode={addNode}
                applyTemplate={applyTemplate}
                toggleZenMode={() => {
                    const newState = !isLeftSidebarOpen;
                    setIsLeftSidebarOpen(newState);
                    setIsRightSidebarOpen(newState);
                }}
                exportAsImage={() => exportAsImage(null)}
            />
        }
       <button
            onClick={() => setIsLeftSidebarOpen(!isLeftSidebarOpen)}
            className="absolute top-3 z-50 p-2 rounded-md bg-white dark:bg-gray-800/50 hover:bg-gray-100 dark:hover:bg-gray-700 backdrop-blur-sm border border-gray-200 dark:border-gray-700 transition-all duration-300 ease-in-out text-gray-600 dark:text-gray-300"
            style={{ left: isLeftSidebarOpen ? '268px' : '12px' }}
            title={isLeftSidebarOpen ? "Hide Menu" : "Show Menu"}
        >
            <MenuIcon className="w-6 h-6" />
        </button>
      <LeftSidebar 
        onExportJson={exportAsJson}
        onExportImage={() => exportAsImage(null)}
        onSelectAreaForExport={() => setIsSelectingForExport(true)}
        onExportGif={handleExportGif}
        isRecordingGif={isRecordingGif}
        isZenMode={isZenMode} // Keeping prop for now but handled via isOpen
        isOpen={isLeftSidebarOpen}
        snapshots={snapshots}
        onTakeSnapshot={(name) => {
            setSnapshots(prev => [...prev, {id: Date.now().toString(), name, state: getCurrentState()}]);
            addToast('Snapshot saved!');
        }}
        onRestoreSnapshot={(id) => {
            const snapshot = snapshots.find(s => s.id === id);
            if(snapshot) {
                pushToUndoStack();
                loadState(snapshot.state);
                addToast(`Restored snapshot: ${snapshot.name}`);
            }
        }}
        onDeleteSnapshot={(id) => setSnapshots(prev => prev.filter(s => s.id !== id))}
        onApplyTemplate={(template) => applyTemplate(template, { x: 200, y: 200 })}
        theme={theme}
        toggleTheme={() => setTheme(t => t === 'light' ? 'dark' : 'light')}
        appMode={appMode}
        setAppMode={handleModeChange}
        onClose={() => setIsLeftSidebarOpen(false)}
        onAddCategory={handleAddCategory}
      />
      <main className="flex-1 relative overflow-hidden transition-all duration-300">
        <AlignmentToolbar />
        <div className="absolute bottom-4 left-4 z-50 flex items-center space-x-3">
            {/* Edit Actions */}
            <div className="flex items-center space-x-1 bg-white dark:bg-gray-800 p-1.5 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700/50">
                <button 
                    title="Select Mode" 
                    onClick={() => setIsMultiSelectMode(!isMultiSelectMode)} 
                    className={`p-2 rounded-md transition-colors ${isMultiSelectMode ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400' : 'text-gray-500 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'}`}
                >
                    <CursorIcon className="w-5 h-5"/>
                </button>
                <div className="w-px h-6 bg-gray-300 dark:bg-gray-600"></div>
                <button title="Erase Canvas" onClick={handleErase} className="p-2 rounded-md text-gray-500 dark:text-gray-300 hover:bg-red-500 hover:text-white transition-colors"><EraseIcon className="w-5 h-5"/></button>
                <div className="w-px h-6 bg-gray-300 dark:bg-gray-600"></div>
                <button title="Undo (Ctrl+Z)" onClick={handleUndo} disabled={undoStack.length === 0} className="p-2 rounded-md text-gray-500 dark:text-gray-300 hover:bg-orange-500 hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent"><UndoIcon className="w-5 h-5"/></button>
                <button title="Redo (Ctrl+Shift+Z)" onClick={handleRedo} disabled={redoStack.length === 0} className="p-2 rounded-md text-gray-500 dark:text-gray-300 hover:bg-orange-500 hover:text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-transparent"><RedoIcon className="w-5 h-5"/></button>
                 <div className="w-px h-6 bg-gray-300 dark:bg-gray-600"></div>
                 {/* Zoom Controls */}
                 <button title="Zoom Out" onClick={() => handleZoom(-0.1)} className="p-2 rounded-md text-gray-500 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"><ZoomOutIcon className="w-5 h-5"/></button>
                 <div className="relative group flex items-center">
                    <input 
                        type="number" 
                        value={Math.round(transform.k * 100)} 
                        onChange={(e) => setZoomLevel(parseInt(e.target.value) || 100)}
                        className="w-10 text-center bg-transparent text-sm font-bold text-gray-600 dark:text-gray-300 focus:outline-none appearance-none m-0 p-0 border-none"
                        min="10" max="500"
                    />
                    <span className="text-xs text-gray-400 font-medium">%</span>
                 </div>
                 <button title="Zoom In" onClick={() => handleZoom(0.1)} className="p-2 rounded-md text-gray-500 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"><ZoomInIcon className="w-5 h-5"/></button>
            </div>
        </div>
        
        {/* Error Validation Panel (AWS Mode Only) */}
        {selectedIssue && appMode === 'aws' && (
            <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2 z-50 bg-white dark:bg-gray-800 rounded-lg shadow-2xl border-l-4 border-red-500 p-4 max-w-lg w-full animate-bounce-in">
                <div className="flex justify-between items-start">
                    <div>
                        <h3 className="text-sm font-bold text-red-600 dark:text-red-400 uppercase tracking-wide mb-1">
                            Configuration Error
                        </h3>
                        <p className="text-sm text-gray-700 dark:text-gray-300 font-medium">
                            {selectedIssue.message}
                        </p>
                        {selectedIssue.fixAction && (
                            <p className="text-xs text-gray-500 mt-1">
                                {selectedIssue.fixAction.position === 'before' ? 'Insert' : 'Append'} {selectedIssue.fixAction.label} to fix connectivity.
                            </p>
                        )}
                    </div>
                </div>
                <div className="mt-4 flex gap-3">
                    {selectedIssue.fixAction && (
                        <button
                            onClick={() => handleAutoFix(selectedIssue!)}
                            className="bg-red-600 hover:bg-red-700 text-white px-4 py-1.5 rounded text-sm font-bold transition-colors shadow-sm"
                        >
                            Fix: {selectedIssue.fixAction.label}
                        </button>
                    )}
                    <button
                        onClick={() => setSelectedIssue(null)}
                        className="bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 px-4 py-1.5 rounded text-sm font-medium transition-colors"
                    >
                        Dismiss
                    </button>
                </div>
            </div>
        )}

        <Canvas
          nodes={nodes}
          connections={connections}
          groups={groups}
          shapes={shapes}
          textNodes={textNodes}
          onAddNode={(service, pos) => addNode(service, pos)}
          onAddGroup={addGroup}
          onAddShape={addShape}
          onAddTextNode={addTextNode}
          onAddRelatedNode={addRelatedNode}
          onUpdateNodePosition={updateNodePosition}
          onAddConnection={addConnection}
          onUpdateGroupPosition={updateGroupPosition}
          onUpdateGroup={updateGroup}
          onUpdateShape={updateShape}
          onUpdateTextNode={updateTextNode}
          onUpdateNodeData={updateNodeData}
          onDeleteNode={deleteNode}
          onDisconnectNode={disconnectNode}
          onStartWorkflow={handleStartWorkflowWrapper}
          onStopWorkflow={handleStopWorkflow}
          workflowRunningNodeId={workflowRunningNodeId} // Kept for legacy compatibility if needed
          selection={selection}
          onSelect={onSelect}
          isSelectingForExport={isSelectingForExport}
          isMultiSelectMode={isMultiSelectMode}
          onExportArea={exportAsImage}
          onMultiSelect={handleMultiSelect}
          onCancelExportSelection={() => setIsSelectingForExport(false)}
          animationState={animationState}
          onInteractionEnd={pushToUndoStack}
          transform={transform}
          setTransform={handleManualTransformChange}
          // Pass down props for tutorial highlighting if needed, or handle in DraggableNode logic based on ID
          activeTutorialNodeId={tutorialState.highlightNodeId}
          cameraTransitionDuration={cameraTransitionDuration}
        />
      </main>
      <RightSidebar 
        selection={selection}
        node={selectedNode || null}
        compareNode={compareNode}
        connection={selectedConnection || null}
        group={selectedGroup || null}
        shape={selectedShape || null}
        textNode={selectedTextNode || null}
        onUpdateNode={updateNodeData}
        onDeleteNode={deleteNode}
        onDisconnectNode={disconnectNode} 
        onUpdateConnection={updateConnectionData}
        onDeleteConnection={deleteConnection}
        onUpdateGroup={updateGroupData}
        onDeleteGroup={deleteGroup}
        onUpdateShape={updateShapeData}
        onDeleteShape={deleteShape}
        onUpdateTextNode={updateTextNodeData}
        onDeleteTextNode={deleteTextNode}
        onDeselect={() => setSelection([])}
        onStartWorkflow={handleToggleWorkflow}
        onStopWorkflow={handleStopWorkflow}
        workflowStatus={workflowStatus}
        onAlign={handleAlign}
        isZenMode={isZenMode} // Pass false or logic for width
        isOpen={isRightSidebarOpen} // Pass visibility state
        onClose={() => setIsRightSidebarOpen(false)} // Pass close handler
        compareMode={compareMode}
        setCompareMode={setCompareMode}
        nodes={nodes}
        validationIssues={validationIssues}
        appMode={appMode} // Pass mode to hide TF button in ML
        tutorialGlowButton={tutorialState.glowButton}
        tutorialMessage={tutorialState.message}
        onPauseWorkflow={() => {
            if (workflowStatus === 'running') {
                setWorkflowStatus('paused');
                addToast('Paused on interaction', 'bg-yellow-500');
            }
        }}
      />
    </div>
  );
};

export default App;
