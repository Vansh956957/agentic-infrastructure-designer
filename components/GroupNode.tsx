
import React, { useState, useCallback, useRef, MouseEvent as ReactMouseEvent } from 'react';
import { Group, Vector2D, GroupData } from '../types';
import { LockIcon } from './Icons';

interface GroupNodeProps {
    group: Group;
    onMove: (groupId: string, delta: Vector2D) => void;
    onUpdate: (groupId: string, data: Partial<GroupData> & {size?: Group['size'], position?: Group['position']}) => void;
    onSelect: (groupId: string, isMulti: boolean) => void;
    isSelected: boolean;
    transformScale: number;
    onInteractionEnd: () => void;
}

const ResizeHandle: React.FC<{ onMouseDown: (e: ReactMouseEvent) => void, cursor: string, style: React.CSSProperties }> = ({ onMouseDown, cursor, style }) => (
    <div
        className="resize-handle"
        style={{ ...style, cursor }}
        onMouseDown={onMouseDown}
    />
);

export const GroupNode: React.FC<GroupNodeProps> = ({ group, onMove, onUpdate, onSelect, isSelected, transformScale, onInteractionEnd }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [isResizing, setIsResizing] = useState<string | null>(null);
    const dragStartPos = useRef({ x: 0, y: 0 });
    const groupRef = useRef<HTMLDivElement>(null);

    const handleMouseDown = useCallback((e: ReactMouseEvent) => {
        if (e.button !== 0 || group.data.locked) return;
        e.stopPropagation();
        onSelect(group.id, e.shiftKey);
        setIsDragging(true);
        dragStartPos.current = { x: e.clientX, y: e.clientY };
    }, [group.id, onSelect, group.data.locked]);

    const handleMouseUp = useCallback(() => {
        if (isDragging || isResizing) {
            onInteractionEnd();
        }
        setIsDragging(false);
        setIsResizing(null);
        document.body.style.cursor = 'default';
    }, [isDragging, isResizing, onInteractionEnd]);

    const handleMouseMove = useCallback((e: MouseEvent) => {
        if (isDragging) {
            const delta = {
                x: (e.clientX - dragStartPos.current.x) / transformScale,
                y: (e.clientY - dragStartPos.current.y) / transformScale,
            };
            onMove(group.id, delta);
            dragStartPos.current = { x: e.clientX, y: e.clientY };
        } else if (isResizing) {
            const newPos = { ...group.position };
            const newSize = { ...group.size };
            const dx = (e.clientX - dragStartPos.current.x) / transformScale;
            const dy = (e.clientY - dragStartPos.current.y) / transformScale;

            if (isResizing.includes('s')) newSize.height = Math.max(100, group.size.height + dy);
            if (isResizing.includes('n')) {
                newSize.height = Math.max(100, group.size.height - dy);
                newPos.y = group.position.y + dy;
            }
            if (isResizing.includes('e')) newSize.width = Math.max(150, group.size.width + dx);
            if (isResizing.includes('w')) {
                newSize.width = Math.max(150, group.size.width - dx);
                newPos.x = group.position.x + dx;
            }
            onUpdate(group.id, { position: newPos, size: newSize });
            dragStartPos.current = { x: e.clientX, y: e.clientY };
        }
    }, [isDragging, isResizing, group, onMove, onUpdate, transformScale]);

    React.useEffect(() => {
        if (isDragging || isResizing) {
            window.addEventListener('mousemove', handleMouseMove);
            window.addEventListener('mouseup', handleMouseUp);
        } else {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        }
        return () => {
            window.removeEventListener('mousemove', handleMouseMove);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isDragging, isResizing, handleMouseMove, handleMouseUp]);
    
    const handleResizeStart = (e: ReactMouseEvent, cursor: string) => {
        e.stopPropagation();
        if (group.data.locked) return;
        setIsResizing(cursor);
        dragStartPos.current = { x: e.clientX, y: e.clientY };
        document.body.style.cursor = `${cursor}-resize`;
    };

    const borderStyle = {
        border: `2px ${group.data.borderStyle} ${isSelected ? '#f90' : group.data.borderColor}`,
        backgroundColor: group.data.backgroundColor,
    };
    
    const handles = [
        { pos: 'n', style: { top: -5, left: '50%', transform: 'translateX(-50%)' } },
        { pos: 's', style: { bottom: -5, left: '50%', transform: 'translateX(-50%)' } },
        { pos: 'w', style: { left: -5, top: '50%', transform: 'translateY(-50%)' } },
        { pos: 'e', style: { right: -5, top: '50%', transform: 'translateY(-50%)' } },
        { pos: 'nw', style: { top: -5, left: -5 } },
        { pos: 'ne', style: { top: -5, right: -5 } },
        { pos: 'sw', style: { bottom: -5, left: -5 } },
        { pos: 'se', style: { bottom: -5, right: -5 } },
    ];

    return (
        <div
            ref={groupRef}
            className="absolute rounded-lg transition-all duration-100 ease-out"
            style={{
                left: group.position.x,
                top: group.position.y,
                width: group.size.width,
                height: group.size.height,
                cursor: group.data.locked ? 'default' : 'move',
                ...borderStyle,
                zIndex: group.zIndex,
            }}
            onMouseDown={handleMouseDown}
        >
            <div className="group-label" style={{ borderColor: isSelected ? '#f90' : group.data.borderColor, color: isSelected ? '#f90' : 'white' }}>
                {group.data.locked && <LockIcon className="w-3 h-3 inline-block mr-1" />}
                {group.data.label}
            </div>
            {!group.data.locked && isSelected && handles.map(h => (
                <ResizeHandle key={h.pos} cursor={`${h.pos}-resize`} style={h.style} onMouseDown={e => handleResizeStart(e, h.pos)} />
            ))}
        </div>
    );
};
