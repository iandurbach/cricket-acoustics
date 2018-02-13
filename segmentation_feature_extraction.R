require(tuneR)
require(seewave)
require(signal)
require(gtools)
require(dtt)
require(rgl)

#function to compute STFT. 
#wav == wave file
#ovlp == amount of overlapy between consercutive frame
#Fs == sampling frequency
#N == number of samples in each window
#nFFT == number of FFT coeffiencts
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

#traces peaks of a vector x between
# two frequency bands low and high
findPeak = function(x, low, high)
{
  y = 20*log10(abs(x[low:high, ]))
  peak = apply(y, 2, max)
  return(peak)
}

#function to 'pad' frames during the segmentation process
padFrames  = function(stft, mat, preFram, postFram)
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
  
  if(mat[n,2] > (ncol(stft$amp)-postFram) )
  {
    matt[n,2] = ncol(stft$amp)
  }else
  {
    matt[n,2] = mat[n,2] + postFram
  }
  
  return(matt)
}

#function used in the segmentation process. 
traceFunc = function(x, thresh, trac, merg, minSamples)
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
  sData = data[order(data[,1]),]
  delete = track = c()
  for(r in 1:(nrow(sData)-1))
  {
    if(sData[r,2] > sData[(r+1),1] || sData[r,2]>sData[(r+1),2])
    {
      delete = c(delete, r)
    }
  }
  if(!is.null(delete))
  {
    sData = sData[-c(delete),]
  }
  for(ss in 1:(nrow(sData)-1) )
  {
    if( abs((sData[ss,2] - sData[(ss+1),1])) < merg )
    {
      track = rbind(track, c(ss, (ss+1)) )
    }
  }
  
  #check for consecutive rows that meet the criteria
  grp = split(track, cumsum(c(0, diff(track[,2])!=1)))
  ssData = sData
  for(tt in  1:length(grp))
  { 
    vec = unique(grp[[tt]])
    ssData[c(vec),1] = sData[min(vec),1]
    ssData[c(vec),2] = sData[max(vec),2]
  }
  ssData = ssData[!duplicated(ssData), ]
  
  delete = c()
  for(r in 1:(nrow(ssData)))
  {
    if(length(seq(ssData[r,1]:ssData[r,2])) < minSamples )
    {
      delete = c(delete, r)
    }
  }
  if(!is.null(delete))
  {
    ssData = ssData[-c(delete),]
  }
  return(ssData)
}

#function to segment the signals. It takes a number of parameters,
#which are explained in the research paper
clickDetect = function(x, stft, thresh, trac, preMerge, minSamples, merg, preFram, postFram)
{
  ssData = traceFunc(x, thresh, trac, merg)
  if(!is.null(ssData) && nrow(ssData) >1 )
  {
    sData = padFrames(stft, ssData, preFram, postFram)
    
    newDataFrame = c()
    for(r in 1:(nrow(sData)))
    {
      if(length(seq(sData[r,1]:sData[r,2])) < preMerge )
      {
        newDataFrame = rbind(newDataFrame, sData[r,])
      }
      else
      {
        y = x[sData[r,1]:sData[r,2]]
        newDataFrame = rbind(newDataFrame, (traceFunc(y, thresh, trac, merg)+sData[r,1])) 
      }
    }
    
    newDataFrame = padFrames(stft, newDataFrame, preFram, postFram)
    Time = stft$time 
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

#function to calculate the MFCC
#frames == STFT
#lowerfreq and upperfreq == the minimum and maximum frequencies to consider
#in Herts
#nfilterbank == the number of filterbanks that are required
#Fs == sampling frequency
#nMFCC == number of MFCC feature coefficients to retain
#n window width for the computation of the MFCC derivatives. Ended up not
#using these in the research
calcMFCC = function(frames, lowerfreq, upperfreq, nfilterbank, Fs, nMFCC, n)
{
  nFFT = 2*nrow(frames)
  #convert to mel scale
  C = 1000/log(1+1000/700)
  lowermel = C*log(1+lowerfreq/700)
  uppermel = C*log(1+upperfreq/700)  
  
  cols = abs(frames) #magnitude spectrum
  
  #calculate mel energies 
  filbank = (melfilterbank(f=Fs, wl=nFFT, m=26))$amp
  energ  = log(t(cols)%*%filbank)
  
  #take discrete cosine tranform of the log filterbank energies to get
  mfcc = (dct(energ)[,1:13])*sqrt(26/2)
  
  delt = deltas(t(mfcc), w=n)
  deltdelt = deltas(delt, w=n)
  return(cbind(mfcc, t(delt), t(deltdelt)))
}
#function to extract samples from wave wave after segmentation.
#time is a dataframe of endpoints obtained from segmentation
getclickSamples = function(samples, times, cc)
{
  t = (time(samples)-1)
  st = which.min(abs(t - times[cc,1])) 
  en = which.min(abs(t - times[cc,2]))
  s = samples[st:en]
  return(s)
}



#extract path of sound files
sound_files_paths 

#looping over all sound files to do the following
#1. Segment cricket calls into syllables
#2. Segment criket calls into chirps
#2.1 Store segmented chirp seuquences
#2.2 Compute MFCC from segmented chirps

for(iter in 1:length(sound_files_path))
{
  #read in sound waves
  sound = readWave(paste(nam[iter], sep=""))
  
  #extract vector of time series values
  file = ts(sound@left, frequency = 44100)
  
  #normalise the amplitude
  file = file/max(abs(file))
  
  #compute the short time Fourier transform
  ST = shortTFT(wav=file, ovlp=95, Fs=44100, N=512, nFFT=512)
  
  #get the magnitude spectrum
  ampl = abs(ST$amp)
  
  #find peaks of spectrum in readiness for segmentation algorithm
  peaks = findPeak(ampl, 47, 94)
  
  
  #A. Syllable segmentation
  #thresholds for syllable segmentation
  out = clickDetect(peaks, ST, 18, 2, 62, 13, 0.1, 0, 0)
  
  times = out$times
  times_vec = c(t(times))
  times_real = (time(file)-1)
  windowss = out$windows
  windowss_vec = c(t(windowss))
  
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
    #write temporal features to file
  }
  
  
  #A. Chirp segmentation
  #thresholds for chirp segmentation
  out = clickDetect(peaks, stf, 8, 8, 380, 70, 35, 4, 12)
  
  if(out$detect == "true")
  {
    w = out$windows
    
    for(cc in 1:nrow(w))
    {
      frames = ampl[,seq(w[cc,1], w[cc,2])]
      fundfreq = ff[seq(w[cc,1], w[cc,2]),2]
      mfcc = calcMFCC(frames = frames, magspec=T, lowerfreq = 0, 
                      upperfreq = 22050, nfilterbank = 26, Fs=44100, nMFCC=13, n=2)
      clicks = getclickSamples(samples = file, times = out$times, cc)
      
      #write MFCC and samples data to file
    }
  }
  
}

