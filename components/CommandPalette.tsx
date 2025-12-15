import React, { useState, useEffect, useRef } from 'react';
import { ALL_SERVICES, ARCHITECTURE_TEMPLATES } from '../constants';
import { AWSService, Template } from '../types';
import { CommandIcon } from './Icons';

interface Command {
    id: string;
    name: string;
    category: string;
    icon: React.FC<{className?: string}>;
    action: () => void;
}

interface CommandPaletteProps {
    onClose: () => void;
    addNode: (service: AWSService, position: {x: number, y: number}) => void;
    applyTemplate: (template: Template, position: {x: number, y: number}) => void;
    toggleZenMode: () => void;
    exportAsImage: () => void;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({ onClose, addNode, applyTemplate, toggleZenMode, exportAsImage }) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedIndex, setSelectedIndex] = useState(0);
    const inputRef = useRef<HTMLInputElement>(null);
    const modalRef = useRef<HTMLDivElement>(null);

    const commands: Command[] = [
        ...ALL_SERVICES.filter(s => !['group', 'shape-rectangle', 'shape-ellipse', 'text'].includes(s.id)).map(service => ({
            id: `add-${service.id}`,
            name: `Add ${service.name}`,
            category: 'Components',
            icon: service.icon,
            action: () => addNode(service, { x: 200, y: 200 })
        })),
        ...ARCHITECTURE_TEMPLATES.map(template => ({
            id: `template-${template.name.toLowerCase().replace(/\s+/g, '-')}`,
            name: `Template: ${template.name}`,
            category: 'Templates',
            icon: CommandIcon, // Generic icon for templates
            action: () => applyTemplate(template, { x: 100, y: 100 })
        })),
        { id: 'toggle-zen-mode', name: 'Toggle Zen Mode', category: 'General', icon: CommandIcon, action: toggleZenMode },
        { id: 'export-image', name: 'Export as Image', category: 'General', icon: CommandIcon, action: exportAsImage },
    ];

    const filteredCommands = commands.filter(command =>
        command.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    useEffect(() => {
        inputRef.current?.focus();
        
        const handleClickOutside = (event: MouseEvent) => {
            if (modalRef.current && !modalRef.current.contains(event.target as Node)) {
                onClose();
            }
        };
        document.addEventListener("mousedown", handleClickOutside);
        return () => document.removeEventListener("mousedown", handleClickOutside);
    }, [onClose]);

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                setSelectedIndex(prev => Math.min(prev + 1, filteredCommands.length - 1));
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                setSelectedIndex(prev => Math.max(prev - 1, 0));
            } else if (e.key === 'Enter') {
                e.preventDefault();
                if(filteredCommands[selectedIndex]) {
                    filteredCommands[selectedIndex].action();
                    onClose();
                }
            } else if (e.key === 'Escape') {
                onClose();
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [filteredCommands, selectedIndex, onClose]);
    
    useEffect(() => {
        setSelectedIndex(0);
    }, [searchTerm]);

    return (
        <div className="fixed inset-0 bg-black/50 flex items-start justify-center p-16 z-[1000]">
            <div ref={modalRef} className="w-full max-w-2xl bg-gray-800 rounded-lg shadow-2xl border border-gray-700/50">
                <div className="p-2 border-b border-gray-700">
                    <input
                        ref={inputRef}
                        type="text"
                        placeholder="Search for commands..."
                        value={searchTerm}
                        onChange={(e) => setSearchTerm(e.target.value)}
                        className="w-full bg-transparent text-white placeholder-gray-400 focus:outline-none text-lg px-2"
                    />
                </div>
                <ul className="max-h-96 overflow-y-auto">
                    {filteredCommands.length > 0 ? filteredCommands.map((command, index) => (
                        <li key={command.id}
                            className={`flex items-center space-x-4 p-3 cursor-pointer ${selectedIndex === index ? 'bg-orange-500/50' : 'hover:bg-gray-700/50'}`}
                            onMouseDown={(e) => e.preventDefault()}
                            onClick={() => { command.action(); onClose(); }}
                            onMouseEnter={() => setSelectedIndex(index)}
                        >
                           <command.icon className="w-6 h-6 flex-shrink-0" />
                           <div className="flex-1">
                                <span className="text-white">{command.name}</span>
                           </div>
                           <span className="text-xs text-gray-400">{command.category}</span>
                        </li>
                    )) : (
                        <li className="p-4 text-center text-gray-500">No commands found.</li>
                    )}
                </ul>
            </div>
        </div>
    );
};
