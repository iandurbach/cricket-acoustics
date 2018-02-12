require(tuneR)
require(seewave)
require(signal)
require(gtools)

shortTFT = function(wav, ovlp, Fs, N, nFFT)
{
  stp = N - (ovlp/100)*N
  hamm = hamming(N)
  n = length(wav)
  r = ceiling((nFFT)/2)
  c = 1+trunc((n-N)/stp)
  
  data = matrix(0, r, c)
  count = 0; col = 1
  while((count + N) <= n)
  {
    x_win = wav[(count+1):(count+N)]*hamm
    x_fft = fft(x_win)[1:nFFT]
    
    #update the stft matrix
    data[,col] = x_fft[1:r]
    
    #update the indexes
    count = count + stp
    col = col + 1
  }
  
  #calculate the time and frequency vectors
  t = seq(from = N/2, to = N/2+(c-1)*stp, by = stp)/Fs
  freq = c(0:(r-1))*(Fs/nFFT)/1000
  
  list(time = t, freq = freq, amp = data)
}
findPeak = function(x, low, high)
{
  y = 20*log10(abs(x[low:high, ]))
  peak = apply(y, 2, max)
  return(peak)
}
padFrames  = function(ST, mat, preFram, postFram)
{
  n  = nrow(mat);matt = matrix(0, n, 2)
  matt[1,2] = mat[1,2] + postFram
  matt[n,1] = mat[n,1] - preFram
  
  for(i in 2:(n-1))
  {
    matt[i,1] = mat[i,1] - preFram
    matt[i,2] = mat[i,2] + postFram
  }
  
  if(mat[1,1] < preFram)
  {
    matt[1,1] = 1
  }else
  {
    matt[1,1] = mat[1,1] - preFram
  }
  
  if(mat[n,2] > (ncol(ST$amp)-postFram) )
  {
    matt[n,2] = ncol(ST$amp)
  }else
  {
    matt[n,2] = mat[n,2] + postFram
  }
  
  return(matt)
}
traceFunc = function(x, thresh, trac, merg)
{
  fin = beg = c(); status = T
  while(status)
  {
    m_i = which.max(x)
    if(x[m_i] > thresh)
    {
      for(i in m_i:1) #keep track of all xs. if zero, then leave
      {
        if( (x[m_i] - trac) > x[i] )
        {
          m_s = i
          beg = c(beg, m_s)
          break
        }
        if(i==1)
        {
          m_s = i
          beg = c(beg, m_s)
          break
        }
      }
      for(ii in m_i:length(x))
      {
        if((x[m_i] - trac) > x[ii] )
        {
          m_e = ii
          fin = c(fin, m_e)
          break
        }
        if(ii == length(x))
        {
          m_e = ii
          fin = c(fin, m_e)
          break
        }
      }
      x[m_s:m_e] = 0
      #update status
      status = !(all(x <= thresh))
    }else{
      break}
  }
  data = cbind(beg, fin)
  if(is.null(data))
  {
    df = c(x[1],x[length(x)])
  }
  else
  {
    myData = data[order(data[,1]),]
    delete = track = c()
    for(r in 1:(nrow(myData)-1))
    {
      if(myData[r,2] > myData[(r+1),1] || myData[r,2]>myData[(r+1),2])
      {
        delete = c(delete, r)
      }
    }
    if(!is.null(delete))
    {
      myData = myData[-c(delete),]
    }
    for(ss in 1:(nrow(myData)-1) )
    {
      if( abs((myData[ss,2] - myData[(ss+1),1])) < merg )
      {
        track = rbind(track, c(ss, (ss+1)) )
      }
    }
    
    #check for consecutive rows that meet the criteria
    grp = split(track, cumsum(c(0, diff(track[,2])!=1)))
    df = myData
    for(tt in  1:length(grp))
    { 
      vec = unique(grp[[tt]])
      df[c(vec),1] = myData[min(vec),1]
      df[c(vec),2] = myData[max(vec),2]
    }
    df = matrix(df[!duplicated(df), ], ncol=2)
    
    delete = c()
    for(r in 1:(nrow(df)))
    {
      if(length(seq(df[r,1]:df[r,2])) < 13 )
      {
        delete = c(delete, r)
      }
    }
    if(!is.null(delete))
    {
      df = df[-c(delete),]
    }
  }
  return(df)
}

clickDetect = function(x, ST, thresh, trac, merg, preFram, postFram)
{
  ssData = traceFunc(x, thresh, trac, merg)
  if(!is.null(ssData) && nrow(ssData) >1 )
  {
    sData = padFrames(ST, ssData, preFram, postFram)
    
    newDataFrame = c()
    for(r in 1:(nrow(sData)))
    {
      if(length(seq(sData[r,1]:sData[r,2])) <= 62 )
      {
        newDataFrame = rbind(newDataFrame, sData[r,])
      }
      else
      {
        y = x[sData[r,1]:sData[r,2]]
        newDataFrame = rbind(newDataFrame, (traceFunc(y, 18, 2, 0.1)+sData[r,1])) 
      }
    }
    
    newDataFrame = padFrames(ST, newDataFrame, preFram, postFram)
    Time = ST$time 
    times = matrix(0, nrow(newDataFrame), 2)
    for(jj in 1:nrow(newDataFrame))
    {
      times[jj,] = c(Time[newDataFrame[jj,1]],Time[newDataFrame[jj,2]] )
    }
    
    list(detect = "true", n_clicks =nrow(newDataFrame) , times = times, windows = newDataFrame)
  }else
  {
    list(detect="false", n_clicks = 0, times = NA, windows=NA)
  }
  
}


#extract path of sound files
sound_files_paths 


for(iter in 1:length(sound_files_path))
{
  sound = readWave(paste(nam[iter], sep=""))
  
  file = ts(sound@left, frequency = 44100)
  file = file/max(abs(file))
  ST = shortTFT(wav=file, ovlp=95, Fs=44100, N=512, nFFT=512)
  ampl = abs(ST$amp)
  peaks = findPeak(ampl, 47, 94)
  out = clickDetect(peaks, ST, 8, 2, 0.1, 0, 0)
  
  times = out$times
  times_vec = c(t(times))
  times_real = (time(file)-1)
  windowss = out$windows
  windowss_vec = c(t(windowss))
  
  #check = diff(windowss_vec)
  #cals = split(windowss_vec, cumsum(c(0, diff(windowss_vec) > 100) ) )
  tims = split(times_vec, cumsum(c(0, diff(windowss_vec) > 65) ) )
  
  #reassemble into matrices. last two rows will give features.
  Feat = c()
  for(i in 1:length(tims))
  {
    mat = matrix(tims[[i]], ncol=2, byrow = T)
    NumSyllables = nrow(mat)
    
    #compute gaps between syllables.
    if(NumSyllables > 1)
    {
      di = dii = c()
      
      for(j in 1:(NumSyllables -1 ) )
      {
        endPoint1_samp = which.min(abs(times_real - mat[(j+1),1]))
        endPoint2_samp = which.min(abs(times_real - mat[j,2]))
        substract_samp = endPoint1_samp - endPoint2_samp
        di = c(di, substract_samp)
        
        substract_sec = mat[(j+1),1] - mat[j,2]
        dii = c(dii, substract_sec)
      }
      
      #compute length of syllables
      lens = mat[,2] - mat[,1]
      
      #take the average
      avgSyllLen = mean(lens)
      avgInterSyllSec =  mean(dii)
      avgInterSyllSamp = mean(di)
      
      #store in array
      Feat = rbind(Feat, c(NumSyllables,avgSyllLen,avgInterSyllSamp ,avgInterSyllSec))
    }
  }
  
  if(out$detect == "true")
  {
    write.table(Feat, file = paste("TempFeat", iter, ".txt", sep=""), col.names = F, row.names = F )
  }
  print(iter)
}

