function result = conv2PreserveEnergy(data,filter)

result = conv2(data,filter,'same');
rsize = size(result);
fsize = size(filter);
fsize2 = (fsize - 1)/2;
fsum = sum(filter(:));

ystart = 1 + fsize2(1);
yend = rsize(1) - fsize2(1);
xstart = 1 + fsize2(2);
xend = rsize(2) - fsize2(2);

if (rsize(1) > fsize(1))
  for y = 1:fsize2(1)
    % top
    fSumTop = sum(sum(filter(fsize(1)-y:fsize(1),:)));
    if (fSumTop ~= 0)
      result(y,xstart:xend) = result(y,xstart:xend) * fsum / fSumTop;
    end
    
    % bottom
    fSumBottom = sum(sum(filter(1:y,:)));
    if (fSumBottom ~= 0)
      result(rsize(1)-y+1,xstart:xend) = result(rsize(1)-y+1,xstart:xend) * fsum / fSumBottom;
    end
  end
end

if (rsize(2) > rsize(1))
  for x = 1:fsize2(2)
    % left
    fSumLeft = sum(sum(filter(:,fsize(2)-x:fsize(2))));
    if (fSumLeft ~= 0)
      result(ystart:yend,x) = result(ystart:yend,x) * fsum / fSumLeft;
    end
    
    % right
    fSumRight = sum(sum(filter(:,1:x)));
    if (fSumRight ~= 0)
      result(ystart:yend,rsize(2)-x+1) = result(ystart:yend,rsize(2)-x+1) * fsum / fSumRight;
    end
    
    % corners
    if (rsize(1) > fsize(1))
      for y = 1:fsize2(1)
        fSumTL = sum(sum(filter(fsize(1)-y:fsize(1),fsize(2)-x:fsize(2))));
        if (fSumTL ~= 0)
          result(y,x) = result(y,x) * fsum / fSumTL;
        end
        
        fSumTR = sum(sum(filter(fsize(1)-y:fsize(1),1:x)));
        if (fSumTR ~= 0)
          result(y,rsize(2)-x+1) = result(y,rsize(2)-x+1) * fsum / fSumTR;
        end
        
        fSumBL = sum(sum(filter(1:y,fsize(2)-x:fsize(2))));
        if (fSumBL ~= 0)
          result(rsize(1)-y+1,x) = result(rsize(1)-y+1,x) * fsum / fSumBL;
        end
        
        fSumBR = sum(sum(filter(1:y,1:x)));
        if (fSumBR ~= 0)
          result(rsize(1)-y+1,rsize(2)-x+1) = result(rsize(1)-y+1,rsize(2)-x+1) * fsum / fSumBR;
        end
      end
    end
  end
end
