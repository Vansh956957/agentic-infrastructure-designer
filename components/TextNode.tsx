
import React, { useState, useCallback, useRef, MouseEvent as ReactMouseEvent, useEffect } from 'react';
import { TextNode as TextNodeType, Vector2D, TextNodeData } from '../types';

interface TextNodeProps {
    textNode: TextNodeType;
    onUpdate: (textId: string, data: Partial<TextNodeData> & {size?: TextNodeType['size'], position?: TextNodeType['position']}) => void;
    onSelect: (textId: string, isMulti: boolean) => void;
    isSelected: boolean;
    transformScale: number;
    onInteractionEnd: () => void;
}

const ResizeHandle: React.FC<{ onMouseDown: (e: ReactMouseEvent) => void, cursor: string, style: React.CSSProperties }> = ({ onMouseDown, cursor, style }) => (
    <div
        className="resize-handle"
        style={{ ...style, cursor, zIndex: 100 }}
        onMouseDown={onMouseDown}
    />
);

export const TextNode: React.FC<TextNodeProps> = ({ textNode, onUpdate, onSelect, isSelected, transformScale, onInteractionEnd }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [isResizing, setIsResizing] = useState<string | null>(null);
    const [isEditing, setIsEditing] = useState(false);
    const dragStartPos = useRef({ x: 0, y: 0 });
    const dragStartNode = useRef(textNode);
    const textRef = useRef<HTMLDivElement>(null);
    
    const { position, size, data, id } = textNode;

    const handleMouseDown = useCallback((e: ReactMouseEvent) => {
        if (e.button !== 0 || isEditing) return;
        e.stopPropagation();
        onSelect(id, e.shiftKey);
        setIsDragging(true);
        dragStartPos.current = { x: e.clientX, y: e.clientY };
        dragStartNode.current = textNode;
    }, [id, onSelect, textNode, isEditing]);

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
            const dx = (e.clientX - dragStartPos.current.x) / transformScale;
            const dy = (e.clientY - dragStartPos.current.y) / transformScale;
            const newPos = {
                x: dragStartNode.current.position.x + dx,
                y: dragStartNode.current.position.y + dy,
            }
            onUpdate(id, { position: newPos });
        } else if (isResizing) {
            const dx = (e.clientX - dragStartPos.current.x) / transformScale;
            const dy = (e.clientY - dragStartPos.current.y) / transformScale;

            let newPos = { ...dragStartNode.current.position };
            let newSize = { ...dragStartNode.current.size };

            if (isResizing.includes('s')) newSize.height = Math.max(20, dragStartNode.current.size.height + dy);
            if (isResizing.includes('n')) {
                const heightChange = Math.min(dy, dragStartNode.current.size.height - 20);
                newSize.height = dragStartNode.current.size.height - heightChange;
                newPos.y = dragStartNode.current.position.y + heightChange;
            }
            if (isResizing.includes('e')) newSize.width = Math.max(20, dragStartNode.current.size.width + dx);
            if (isResizing.includes('w')) {
                const widthChange = Math.min(dx, dragStartNode.current.size.width - 20);
                newSize.width = dragStartNode.current.size.width - widthChange;
                newPos.x = dragStartNode.current.position.x + widthChange;
            }
            onUpdate(id, { position: newPos, size: newSize });
        }
    }, [isDragging, isResizing, id, onUpdate, transformScale]);

    useEffect(() => {
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
        setIsResizing(cursor);
        dragStartPos.current = { x: e.clientX, y: e.clientY };
        dragStartNode.current = textNode;
        document.body.style.cursor = `${cursor}-resize`;
    };

    const handleDoubleClick = () => {
        setIsEditing(true);
    }
    
    const handleBlur = () => {
        setIsEditing(false);
        if (textRef.current) {
            if(textRef.current.innerText !== data.content) {
                onInteractionEnd(); // Save history if content changed
                onUpdate(id, { content: textRef.current.innerText });
            }
        }
    }

    useEffect(() => {
        if (isEditing && textRef.current) {
            textRef.current.focus();
            document.execCommand('selectAll', false);
        }
    }, [isEditing]);
    
    const handles = [
        { pos: 'nw', style: { top: -5, left: -5 } },
        { pos: 'ne', style: { top: -5, right: -5 } },
        { pos: 'sw', style: { bottom: -5, left: -5 } },
        { pos: 'se', style: { bottom: -5, right: -5 } },
    ];
    
    return (
        <div
            className={`absolute ${isSelected ? 'p-1' : ''}`}
            style={{
                left: position.x,
                top: position.y,
                width: size.width,
                height: size.height,
                cursor: isEditing ? 'text' : 'move',
                zIndex: textNode.zIndex,
            }}
            onMouseDown={handleMouseDown}
            onDoubleClick={handleDoubleClick}
        >
            <div
                ref={textRef}
                contentEditable={isEditing}
                onBlur={handleBlur}
                suppressContentEditableWarning={true}
                className={`w-full h-full outline-none break-words ${isEditing ? 'text-node-editing' : ''}`}
                style={{
                    fontSize: `${data.fontSize}px`,
                    color: data.color,
                }}
            >
                {data.content}
            </div>
            {isSelected && !isEditing && (
              <div className="absolute top-0 left-0 w-full h-full border-2 border-dashed border-orange-500 pointer-events-none" />
            )}
            {isSelected && !isEditing && handles.map(h => (
                <ResizeHandle key={h.pos} cursor={`${h.pos}-resize`} style={h.style} onMouseDown={e => handleResizeStart(e, h.pos)} />
            ))}
        </div>
    );
};
