use std::{cell::RefCell, path::PathBuf, rc::Rc};

use windows::{
    Win32::{Foundation::S_OK, System::Com::CoTaskMemFree, UI::Shell::*},
    core::{BOOL, Interface, Ref, Result, implement, w},
};

const SELECT_CURRENT_FOLDER: u32 = 1;

/// CDXC:PlatformSupport 2026-09-23 WHY:
/// FOS_PICKFOLDERS hides files even when a caller requests both files and directories. Keep the native file picker in file mode and offer a separate current-folder button so attachments retain both capabilities.
pub(crate) struct MixedPathDialog {
    cookie: u32,
    selected: Rc<RefCell<Option<PathBuf>>>,
}

impl MixedPathDialog {
    pub(crate) fn new(dialog: &IFileOpenDialog) -> Result<Self> {
        let customize: IFileDialogCustomize = dialog.cast()?;
        unsafe { customize.AddPushButton(SELECT_CURRENT_FOLDER, w!("Select current folder"))? };
        let selected = Rc::new(RefCell::new(None));
        let events: IFileDialogEvents = MixedPathEvents {
            selected: selected.clone(),
        }
        .into();
        let cookie = unsafe { dialog.Advise(&events)? };
        Ok(Self { cookie, selected })
    }

    pub(crate) fn finish(self, dialog: &IFileOpenDialog) -> Result<Option<PathBuf>> {
        unsafe { dialog.Unadvise(self.cookie)? };
        Ok(self.selected.borrow_mut().take())
    }
}

#[implement(IFileDialogEvents, IFileDialogControlEvents)]
struct MixedPathEvents {
    selected: Rc<RefCell<Option<PathBuf>>>,
}

#[allow(non_snake_case)]
impl IFileDialogControlEvents_Impl for MixedPathEvents_Impl {
    fn OnButtonClicked(&self, customize: Ref<'_, IFileDialogCustomize>, id: u32) -> Result<()> {
        if id == SELECT_CURRENT_FOLDER {
            let dialog: IFileDialog = customize.ok()?.cast()?;
            let folder = unsafe { dialog.GetFolder()? };
            let name = unsafe { folder.GetDisplayName(SIGDN_FILESYSPATH)? };
            let path = unsafe { name.to_string() };
            unsafe { CoTaskMemFree(Some(name.0.cast())) };
            *self.selected.borrow_mut() = Some(PathBuf::from(path?));
            unsafe { dialog.Close(S_OK)? };
        }
        Ok(())
    }

    fn OnItemSelected(&self, _: Ref<'_, IFileDialogCustomize>, _: u32, _: u32) -> Result<()> {
        Ok(())
    }

    fn OnCheckButtonToggled(
        &self,
        _: Ref<'_, IFileDialogCustomize>,
        _: u32,
        _: BOOL,
    ) -> Result<()> {
        Ok(())
    }

    fn OnControlActivating(&self, _: Ref<'_, IFileDialogCustomize>, _: u32) -> Result<()> {
        Ok(())
    }
}

#[allow(non_snake_case)]
impl IFileDialogEvents_Impl for MixedPathEvents_Impl {
    fn OnFileOk(&self, _: Ref<'_, IFileDialog>) -> Result<()> {
        Ok(())
    }

    fn OnFolderChanging(&self, _: Ref<'_, IFileDialog>, _: Ref<'_, IShellItem>) -> Result<()> {
        Ok(())
    }

    fn OnFolderChange(&self, _: Ref<'_, IFileDialog>) -> Result<()> {
        Ok(())
    }

    fn OnSelectionChange(&self, _: Ref<'_, IFileDialog>) -> Result<()> {
        Ok(())
    }

    fn OnShareViolation(
        &self,
        _: Ref<'_, IFileDialog>,
        _: Ref<'_, IShellItem>,
    ) -> Result<FDE_SHAREVIOLATION_RESPONSE> {
        Ok(FDESVR_DEFAULT)
    }

    fn OnTypeChange(&self, _: Ref<'_, IFileDialog>) -> Result<()> {
        Ok(())
    }

    fn OnOverwrite(
        &self,
        _: Ref<'_, IFileDialog>,
        _: Ref<'_, IShellItem>,
    ) -> Result<FDE_OVERWRITE_RESPONSE> {
        Ok(FDEOR_DEFAULT)
    }
}
