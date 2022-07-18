import os
import subprocess
import numpy as np
from lsst.ts.wep.Utility import FilterType

from lsst.ts.phosim.telescope.TeleFacade import TeleFacade
from lsst.ts.phosim.OpdMetrology import OpdMetrology
from lsst.ts.phosim.utils.Utility import getConfigDir, getPhoSimPath, getAoclcOutputPath


def main(phosimDir):
    # Settings
    outputDir = getAoclcOutputPath()
    outputImgDir = os.path.join(outputDir, "img")
    os.makedirs(outputImgDir, exist_ok=True)

    configDir = getConfigDir()
    cmdSettingFile = os.path.join(configDir, "cmdFile", "opdDefault.cmd")
    instSettingFile = os.path.join(configDir, "instFile", "opdDefault.inst")

    # Declare the opd metrology and add the interested field points
    metr = OpdMetrology()
    metr.setWgtAndFieldXyOfGQ("lsst")
    wfsXY = metr.getDefaultLsstWfsGQ()
    metr.addFieldXYbyDeg(*wfsXY)

    # Loop over survey parameters
    for zenith_angle, rotation_angle in [
        (0, 0),
        (45, 0),
        (45, 45),
        (30, -30)
    ]:
        for m1m3, m2, camera, feaname in [
            (True, False, False, "M1M3"),
            (False, True, False, "M2"),
            (False, False, True, "Cam"),
            (True, True, True, "All")
        ]:
            name = f"z{zenith_angle}_r{rotation_angle}_{feaname}"

            tele = TeleFacade()
            tele.setPhoSimDir(phosimDir)
            tele.addSubSys(addM1M3=m1m3, addM2=m2, addCam=camera)

            obsId = 9006050
            filterType = FilterType.REF
            tele.setSurveyParam(
                obsId=obsId,
                filterType=filterType,
                zAngleInDeg=zenith_angle,
                rotAngInDeg=rotation_angle,
            )

            pertCmdFilePath = tele.writePertBaseOnConfigFile(
                outputDir, saveResMapFig=True
            )
            # Write the physical command file
            cmdFilePath = tele.writeCmdFile(
                outputDir,
                cmdSettingFile=cmdSettingFile,
                pertFilePath=pertCmdFilePath,
                cmdFileName="opd.cmd"
            )

            # Write the instance file
            instFilePath = tele.writeOpdInstFile(
                outputDir, metr, instSettingFile=instSettingFile, instFileName="opd.inst"
            )

            # Get the argument to run the PhoSim
            logFilePath = None #os.path.join(outputImgDir, "opdPhoSim.log")
            argString = tele.getPhoSimArgs(
                instFilePath,
                extraCommandFile=cmdFilePath,
                numPro=2,
                outputDir=outputImgDir,
                e2ADC=0,
                logFilePath=logFilePath,
            )
            # Run the PhoSim
            tele.runPhoSim(argString)

            # Copy from work dir to out dir
            odir = os.path.join(outputDir, "..", "fea", name)
            os.makedirs(odir, exist_ok=True)
            for iopd in range(35):
                ifn = f"opd_9006050_{iopd}.fits.gz"
                ofn = ifn.replace("9006050", name)
                cmd = [
                    "cp",
                    os.path.join(outputImgDir, ifn),
                    os.path.join(odir, ofn)
                ]
                subprocess.call(" ".join(cmd), shell=True)


if __name__ == "__main__":
    phosimDir = getPhoSimPath()
    main(phosimDir)
