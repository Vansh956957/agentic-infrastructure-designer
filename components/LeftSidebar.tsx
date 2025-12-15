
import React, { useState } from 'react';
import { AWS_CATEGORIES, AI_CATEGORIES, ARCHITECTURE_TEMPLATES, AI_TEMPLATES, DL_MODE_CATEGORIES } from '../constants';
import { AWSService, Snapshot, Template, AWSCategory } from '../types';
import { SunIcon, MoonIcon, XIcon, GroupIcon } from './Icons';

interface LeftSidebarProps {
  onExportJson: () => void;
  onExportImage: () => void;
  onSelectAreaForExport: () => void;
  onExportGif: () => void;
  isRecordingGif: boolean;
  isOpen: boolean;
  isZenMode: boolean;
  snapshots: Snapshot[];
  onTakeSnapshot: (name: string) => void;
  onRestoreSnapshot: (id: string) => void;
  onDeleteSnapshot: (id: string) => void;
  onApplyTemplate: (template: Template) => void;
  theme: 'light' | 'dark';
  toggleTheme: () => void;
  appMode: 'aws' | 'ai' | 'dl';
  setAppMode: (mode: 'aws' | 'ai' | 'dl') => void;
  onClose: () => void;
  onAddCategory: (category: AWSCategory) => void;
}

const DraggableServiceItem: React.FC<{ service: AWSService }> = ({ service }) => {
  const handleDragStart = (e: React.DragEvent) => {
    e.dataTransfer.setData('application/aws-service', service.id);
  };

  return (
    <div className="tooltip group">
      <div
        draggable
        onDragStart={handleDragStart}
        className="flex items-center justify-center p-2 rounded-md bg-gray-100 dark:bg-gray-700/50 hover:bg-white dark:hover:bg-gray-700 hover:shadow-md cursor-grab transition-all border border-gray-200 dark:border-gray-600"
      >
        <service.icon className="w-8 h-8 text-gray-600 dark:text-gray-300 group-hover:text-gray-800 dark:group-hover:text-white transition-colors" />
      </div>
      <span className="tooltiptext">{service.name}</span>
    </div>
  );
};

const Section: React.FC<{ name: string; children: React.ReactNode, defaultOpen?: boolean, onAddAll?: () => void }> = ({ name, children, defaultOpen = false, onAddAll }) => {
    const [isOpen, setIsOpen] = useState(defaultOpen);
    return (
        <div>
            <div className="flex items-center justify-between mb-2 px-1">
                <button onClick={() => setIsOpen(!isOpen)} className="flex-1 flex justify-between items-center font-semibold text-gray-500 dark:text-gray-400 text-xs uppercase tracking-wider focus:outline-none hover:text-gray-700 dark:hover:text-gray-300">
                    {name}
                    <svg className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path></svg>
                </button>
                {onAddAll && (
                    <button onClick={onAddAll} title="Add all to canvas" className="ml-2 text-gray-400 hover:text-orange-500 dark:hover:text-orange-400 transition-colors">
                        <GroupIcon className="w-4 h-4" />
                    </button>
                )}
            </div>
            {isOpen && children}
        </div>
    );
};


export const LeftSidebar: React.FC<LeftSidebarProps> = ({ 
    onExportJson, onExportImage, onSelectAreaForExport, onExportGif, isRecordingGif, isOpen, isZenMode,
    snapshots, onTakeSnapshot, onRestoreSnapshot, onDeleteSnapshot,
    onApplyTemplate, theme, toggleTheme, appMode, setAppMode, onClose, onAddCategory
 }) => {
  const [exportMode, setExportMode] = useState<'json' | 'gif'>('json');
  const [snapshotName, setSnapshotName] = useState('');
  
  let currentCategories = AWS_CATEGORIES;
  let currentTemplates = ARCHITECTURE_TEMPLATES;

  if (appMode === 'ai') {
      currentCategories = AI_CATEGORIES;
      currentTemplates = AI_TEMPLATES;
  } else if (appMode === 'dl') {
      currentCategories = DL_MODE_CATEGORIES;
      currentTemplates = AI_TEMPLATES; // Reusing AI templates or could define specific DL ones
  }
    
  return (
    <aside className={`bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700/50 flex flex-col shadow-lg transition-all duration-300 ease-in-out overflow-hidden ${!isOpen ? 'w-0 p-0 border-none opacity-0' : 'w-64 p-3 opacity-100'}`}>
      <div className="flex items-start justify-between mb-4 min-w-[200px]">
        <div>
            <h1 className="text-xl font-bold text-gray-800 dark:text-white tracking-wide">
                {appMode === 'aws' ? 'AWS Visualizer' : (appMode === 'dl' ? 'Deep Learning' : 'AI Visualizer')}
            </h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">Designed by Vansh Rewaskar</p>
        </div>
        <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 transition-colors p-1" title="Close Panel">
            <XIcon className="w-5 h-5" />
        </button>
      </div>
      
      <div className="flex items-center space-x-1 mb-4">
             <div className="flex-1 flex bg-gray-100 dark:bg-gray-700 rounded p-0.5">
                 <button onClick={() => setAppMode('aws')} className={`flex-1 text-[10px] font-bold uppercase rounded py-1 ${appMode === 'aws' ? 'bg-white dark:bg-gray-600 shadow text-orange-600 dark:text-white' : 'text-gray-500 dark:text-gray-400'}`}>AWS</button>
                 <button onClick={() => setAppMode('ai')} className={`flex-1 text-[10px] font-bold uppercase rounded py-1 ${appMode === 'ai' ? 'bg-white dark:bg-gray-600 shadow text-blue-600 dark:text-white' : 'text-gray-500 dark:text-gray-400'}`}>ML</button>
                 <button onClick={() => setAppMode('dl')} className={`flex-1 text-[10px] font-bold uppercase rounded py-1 ${appMode === 'dl' ? 'bg-white dark:bg-gray-600 shadow text-purple-600 dark:text-white' : 'text-gray-500 dark:text-gray-400'}`}>DL</button>
             </div>
             <button onClick={toggleTheme} className="p-2 rounded-full hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-500 dark:text-gray-400 transition-colors">
                {theme === 'light' ? <MoonIcon className="w-5 h-5" /> : <SunIcon className="w-5 h-5" />}
            </button>
        </div>

      <div className="flex-1 overflow-y-auto pr-1 space-y-6 scrollbar-thin scrollbar-thumb-gray-300 dark:scrollbar-thumb-gray-600 min-w-[200px]">
            {/* Main Components Section */}
            <div>
                <h3 className="text-xs font-bold text-gray-400 dark:text-gray-500 uppercase tracking-widest mb-3 px-1">Components</h3>
                {currentCategories.map((category) => (
                    <div key={category.name} className="mb-4 last:mb-0">
                        <Section name={category.name} defaultOpen={false} onAddAll={() => onAddCategory(category)}>
                            <div className="grid grid-cols-3 gap-2">
                                {category.services.map((service) => (
                                    <DraggableServiceItem key={service.id} service={service} />
                                ))}
                            </div>
                        </Section>
                    </div>
                ))}
            </div>
            
            <Section name="Templates">
                <div className="space-y-2">
                    {currentTemplates.map((template: Template) => (
                        <button key={template.name} onClick={() => onApplyTemplate(template)}
                            className="w-full text-left text-sm p-2 rounded-md bg-gray-50 dark:bg-gray-700/50 hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 transition-colors border border-transparent hover:border-gray-200 dark:hover:border-gray-600"
                            title={template.description}
                        >
                            {template.name}
                        </button>
                    ))}
                </div>
            </Section>

            <Section name="Snapshots">
                <div className="space-y-2">
                    <div className="flex space-x-2">
                        <input type="text" value={snapshotName} onChange={e => setSnapshotName(e.target.value)} placeholder="Name..."
                            className="flex-1 bg-gray-50 dark:bg-gray-700 border border-gray-200 dark:border-gray-600 rounded-md shadow-sm py-1 px-2 text-gray-900 dark:text-white focus:outline-none focus:ring-orange-500 focus:border-orange-500 text-sm placeholder-gray-400" />
                        <button onClick={() => { if(snapshotName) { onTakeSnapshot(snapshotName); setSnapshotName(''); } }}
                            className="bg-blue-600 hover:bg-blue-700 text-white font-bold p-1.5 rounded disabled:bg-gray-300 dark:disabled:bg-gray-600 disabled:cursor-not-allowed text-xs"
                            disabled={!snapshotName}
                        >Save</button>
                    </div>
                    <div className="space-y-1 max-h-32 overflow-y-auto">
                        {snapshots.map(snapshot => (
                            <div key={snapshot.id} className="flex items-center justify-between text-sm p-1.5 rounded-md bg-gray-50 dark:bg-gray-900/50 border border-gray-100 dark:border-gray-700/50 group">
                                <span className="truncate flex-1 text-gray-700 dark:text-gray-300">{snapshot.name}</span>
                                <div className="flex items-center opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button onClick={() => onRestoreSnapshot(snapshot.id)} className="p-1 text-gray-400 hover:text-green-500" title="Load">
                                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 4v5h5M20 20v-5h-5M4 4l16 16"></path></svg>
                                    </button>
                                    <button onClick={() => onDeleteSnapshot(snapshot.id)} className="p-1 text-gray-400 hover:text-red-500" title="Delete">
                                        <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
                                    </button>
                                </div>
                            </div>
                        ))}
                        {snapshots.length === 0 && <p className="text-xs text-gray-400 text-center italic py-2">No snapshots yet</p>}
                    </div>
                </div>
            </Section>
      </div>

      <div className="mt-auto pt-4 border-t border-gray-200 dark:border-gray-700 min-w-[200px]">
        <div className="bg-gray-100 dark:bg-gray-700 rounded-md p-1 flex mb-2 text-sm">
            <button onClick={() => setExportMode('json')} className={`flex-1 text-center py-1 rounded text-xs font-medium ${exportMode === 'json' ? 'bg-white dark:bg-gray-600 text-orange-600 dark:text-white shadow-sm' : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'}`}>JSON</button>
            <button onClick={() => setExportMode('gif')} className={`flex-1 text-center py-1 rounded text-xs font-medium ${exportMode === 'gif' ? 'bg-white dark:bg-gray-600 text-orange-600 dark:text-white shadow-sm' : 'text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200'}`}>GIF</button>
        </div>
        
        {exportMode === 'json' && (
             <button
              onClick={onExportJson}
              className="w-full bg-gray-800 dark:bg-indigo-600 hover:bg-gray-900 dark:hover:bg-indigo-700 text-white font-bold py-2 px-3 rounded transition-colors text-xs"
            >
              Export JSON
            </button>
        )}
        {exportMode === 'gif' && (
            <div className="space-y-2">
              <button
                onClick={onExportGif}
                disabled={isRecordingGif}
                className={`w-full font-bold py-2 px-3 rounded transition-colors text-xs ${isRecordingGif 
                    ? 'bg-red-500 text-white animate-pulse cursor-wait' 
                    : 'bg-gray-800 dark:bg-teal-600 hover:bg-gray-900 dark:hover:bg-teal-700 text-white'}`}
              >
                {isRecordingGif ? 'Recording (15s)...' : 'Record Workflow GIF'}
              </button>
              <p className="text-[10px] text-gray-500 dark:text-gray-400 text-center">Captures 15s of active flow</p>
            </div>
        )}
      </div>
    </aside>
  );
};
