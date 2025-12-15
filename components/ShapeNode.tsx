
import React, { useState, useCallback, useRef, MouseEvent as ReactMouseEvent } from 'react';
import { Shape, Vector2D, ShapeData } from '../types';

interface ShapeNodeProps {
    shape: Shape;
    onUpdate: (shapeId: string, data: Partial<ShapeData> & {size?: Shape['size'], position?: Shape['position']}) => void;
    onSelect: (shapeId: string, isMulti: boolean) => void;
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

export const ShapeNode: React.FC<ShapeNodeProps> = ({ shape, onUpdate, onSelect, isSelected, transformScale, onInteractionEnd }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [isResizing, setIsResizing] = useState<string | null>(null);
    const dragStartPos = useRef({ x: 0, y: 0 });
    const dragStartShape = useRef(shape);
    const shapeRef = useRef<HTMLDivElement>(null);

    const handleMouseDown = useCallback((e: ReactMouseEvent) => {
        if (e.button !== 0) return;
        e.stopPropagation();
        onSelect(shape.id, e.shiftKey);
        setIsDragging(true);
        dragStartPos.current = { x: e.clientX, y: e.clientY };
        dragStartShape.current = shape;
    }, [shape.id, onSelect, shape]);

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
                x: dragStartShape.current.position.x + dx,
                y: dragStartShape.current.position.y + dy,
            }
            onUpdate(shape.id, { position: newPos });
        } else if (isResizing) {
            const dx = (e.clientX - dragStartPos.current.x) / transformScale;
            const dy = (e.clientY - dragStartPos.current.y) / transformScale;

            let newPos = { ...dragStartShape.current.position };
            let newSize = { ...dragStartShape.current.size };

            if (isResizing.includes('s')) newSize.height = Math.max(20, dragStartShape.current.size.height + dy);
            if (isResizing.includes('n')) {
                const heightChange = Math.min(dy, dragStartShape.current.size.height - 20);
                newSize.height = dragStartShape.current.size.height - heightChange;
                newPos.y = dragStartShape.current.position.y + heightChange;
            }
            if (isResizing.includes('e')) newSize.width = Math.max(20, dragStartShape.current.size.width + dx);
            if (isResizing.includes('w')) {
                const widthChange = Math.min(dx, dragStartShape.current.size.width - 20);
                newSize.width = dragStartShape.current.size.width - widthChange;
                newPos.x = dragStartShape.current.position.x + widthChange;
            }
            onUpdate(shape.id, { position: newPos, size: newSize });
        }
    }, [isDragging, isResizing, shape.id, onUpdate, transformScale]);

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
        setIsResizing(cursor);
        dragStartPos.current = { x: e.clientX, y: e.clientY };
        dragStartShape.current = shape;
        document.body.style.cursor = `${cursor}-resize`;
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
    
    const borderDashArray = {
      'solid': 'none',
      'dashed': '8, 8',
      'dotted': '2, 6',
    }[shape.data.borderStyle]

    return (
        <div
            ref={shapeRef}
            className="absolute"
            style={{
                left: shape.position.x,
                top: shape.position.y,
                width: shape.size.width,
                height: shape.size.height,
                cursor: 'move',
                zIndex: shape.zIndex,
            }}
            onMouseDown={handleMouseDown}
        >
            <svg width="100%" height="100%" className="absolute pointer-events-none">
                {shape.type === 'rectangle' && (
                    <rect
                        width="100%"
                        height="100%"
                        fill={shape.data.fillColor}
                        stroke={shape.data.strokeColor}
                        strokeWidth={shape.data.strokeWidth}
                        strokeDasharray={borderDashArray}
                    />
                )}
                {shape.type === 'ellipse' && (
                    <ellipse
                        cx="50%"
                        cy="50%"
                        rx="50%"
                        ry="50%"
                        fill={shape.data.fillColor}
                        stroke={shape.data.strokeColor}
                        strokeWidth={shape.data.strokeWidth}
                        strokeDasharray={borderDashArray}
                    />
                )}
            </svg>
            {isSelected && (
              <div className="absolute top-0 left-0 w-full h-full border-2 border-dashed border-orange-500 pointer-events-none" />
            )}
            {isSelected && handles.map(h => (
                <ResizeHandle key={h.pos} cursor={`${h.pos}-resize`} style={h.style} onMouseDown={e => handleResizeStart(e, h.pos)} />
            ))}
        </div>
    );
};
